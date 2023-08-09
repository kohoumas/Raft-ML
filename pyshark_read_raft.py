#!/usr/bin/python3

import sys
import pyshark
from urllib.parse import urlparse # works with Python3
#from urlparse import urlparse # works with Python2
from joblib import dump, load
import sklearn
import pandas as pd

#from functions import failure_labels, add_suffixes

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs) # Python3
    #print  >> sys.stderr, ' '.join(map(str, args)) # Python2

def oprint(*args, **kwargs):
    print(*args, file=sys.stdout, **kwargs) # Python3
    #print  >> sys.stdout, ' '.join(map(str, args)) # Python2
    sys.stdout.flush()

def bprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs) # Python3
    print(*args, file=sys.stdout, **kwargs) # Python3
    #print  >> sys.stderr, ' '.join(map(str, args)) # Python2
    #print  >> sys.stdout, ' '.join(map(str, args)) # Python2
    sys.stdout.flush()

LIVE = False
NODES = 5

files = []
files.append("raft-output-1.pcap")
intf = 's1-eth1'

stream = {}
stream['message'] = {}
stream['msgapp'] = {}
for i in range(1, NODES+1):
  for j in range(1, NODES+1):
    stream['message'][str(i)+str(j)] = -1
    stream['msgapp' ][str(i)+str(j)] = -1

stream_num_to_name = {}

msgs = []

msg_type = -1
msg_to = -1
msg_from = -1
msg_term = -1
msg_reject = 0

MTU = 1500 #bytes
IP_HEADER_LEN = 20 #bytes
TCP_HEADER_LEN = 32 #bytes
MSS = MTU - IP_HEADER_LEN - TCP_HEADER_LEN #Maximum TCP segmentation size in bytes

# the data of the previous seg with size equal to MSS, per stream.
prev_seg_data = {}
prev_seg_data_len = {}

# The following variables are used for ML
SAMPLE_SIZE = 20 # how many messages will be joined to a sample, used by the ML algorithm

last_leader_term = -1 # the highest term of all heartbeats that have been read
# estimator = load('svm.joblib') 

for file in files:
  cap = pyshark.LiveCapture(interface = intf) if LIVE else pyshark.FileCapture(file)

  pkt_range = cap.sniff_continuously() if LIVE else cap
  for pkt in pkt_range:

    if ('HTTP' in pkt) and ('request_full_uri' in pkt.http.field_names):
      from_ip = pkt.ip.src
      from_id = int(from_ip.split('.')[3])
      to_ip = pkt.ip.dst
      to_id = int(to_ip.split('.')[3])
      url = urlparse(pkt.http.request_full_uri)
      key = None

      if "/raft/stream/message/" in url.path: 
        key = "message"
      elif "/raft/stream/msgapp/" in url.path: 
        key = "msgapp"
      
      if key:
        stream[key][str(from_id) + str(to_id)] = pkt.tcp.stream
        stream_num_to_name[pkt.tcp.stream] = key + str(from_id) + str(to_id)
        eprint("stream ", from_id, "->", to_id, " message num: ", pkt.tcp.stream)

        # for each stream initialize the prev_seg_data with an empty string (no data from previous segments, when stream starts).
        prev_seg_data[pkt.tcp.stream] = ""
        prev_seg_data_len[pkt.tcp.stream] = 0

    elif 'TCP' in pkt:
      if pkt.tcp.stream in stream_num_to_name:

        data = None
        if 'payload' in pkt.tcp.field_names: 
          data = pkt.tcp.payload[:]
        elif 'segment_data' in pkt.tcp.field_names:
          data = pkt.tcp.segment_data[:]
        elif 'HTTP' in pkt:
          data = pkt.http.data_data[:] 

        #tcp_len = int(pkt.tcp.len) # in Mininet-VM environment, there are packets that tcp.len is equal to 'Bytes in flight', that is higher than the len(data)
        tcp_len = (len(data)+1)/3 if data else 0
        tcp_len_of_this_packet = tcp_len

        # if data does not exist or starts with HTTP/1, then continue with the next packet
        if not(data) or data[0:17] == "48:54:54:50:2f:31": 
          continue
        elif tcp_len == MSS : # if data is segmented, then continue with the next packet, and read all data together
          if prev_seg_data[pkt.tcp.stream] != "":
            prev_seg_data[pkt.tcp.stream] = prev_seg_data[pkt.tcp.stream] + ":" + data
          else:
            prev_seg_data[pkt.tcp.stream] = data
          prev_seg_data_len[pkt.tcp.stream] += tcp_len
          eprint("TCP pkt segmented (No. ", pkt.number, ", TCP len. ", tcp_len, ") from ", stream_num_to_name[pkt.tcp.stream], " (Stream. ", pkt.tcp.stream, ")", sep="")
          continue
        elif prev_seg_data[pkt.tcp.stream] != "": # if there are previous segmented packets, join them with the current one.
          data = prev_seg_data[pkt.tcp.stream] + ":" + data
          tcp_len += prev_seg_data_len[pkt.tcp.stream]
          prev_seg_data[pkt.tcp.stream] = ""
          prev_seg_data_len[pkt.tcp.stream] = 0
          eprint("TCP pkt aggre/ted (No. ", pkt.number, ", TCP len. of pkt ", tcp_len_of_this_packet, ", TCP len. of all ", tcp_len, ") from ", stream_num_to_name[pkt.tcp.stream], " (Stream. ", pkt.tcp.stream, ")", sep="")
        else:
          eprint("TCP pkt (No. ", pkt.number, ", TCP len. ", tcp_len, ") from ", stream_num_to_name[pkt.tcp.stream], " (Stream. ", pkt.tcp.stream, ")", sep="")

        counter_msg_groups = 0
        data_print = data # all data of this packet, to be printed in case of exception
        while tcp_len:
          try:
            data_len = 0
            tcp_len_sub = 0
            while (data[0:5]!='0d:0a'): # reads the Content-Length field (all bytes until \r\n)
              data_len = 16 * data_len
              if (data[0] == '3'): # this byte corresponds to [0-9]
                data_len = data_len + (ord(data[1]) - 48) # reads [0-9]
              elif (data[0] == '6'): # this byte corresponds to [a-f]
                data_len = data_len + (ord(data[1]) - 39) # reads [a-f]
              data = data[3:] ; tcp_len_sub += 1 # removes the outer byte from the Content-Length field 
            data = data[6:] ; tcp_len_sub += 2 # removes \r\n
            tcp_len -= tcp_len_sub

            counter_msg_groups += 1
            eprint("  ", counter_msg_groups, ". remaining TCP len: (", tcp_len_sub, "+", tcp_len, "), data length: ", data_len, " ", hex(data_len), sep="")

            counter_msg = 0        
            msgapp_type = -1 # -1 (=message from message stream, not msgapp stream)
            while data_len:
              counter_msg += 1
              eprint("    ", counter_msg, ". remaining TCP len: ", tcp_len, ", remaining data length: ", data_len, " ", hex(data_len), sep="")

              if pkt.tcp.stream in list(stream["msgapp"].values()): # for msgapp streams
                msgapp_type = int(data[1]) # 0 (=LinkHeartbeat) or 1 (=MsgAppEntries) or 2 (=MsgApp)
                eprint("      msgapp-type: ", msgapp_type)
                data = data[3:] ; data_len -= 1 ; tcp_len -= 1 # removes MsgAppType
                if msgapp_type == 1: # reads MsgAppEntries
                  msg_type = -3
                  data_len = 0; tcp_len = 0 #ignores the messages following a MsgAppEntry message, which exist in the same packet
                elif msgapp_type == 0: # reads LinkHeartbeats
                  msg_type = -2 # sets msg type to -2 , for LinkHeartbeats in a msgApp stream

              if msgapp_type == 2 or msgapp_type == -1: # MsgApp from a msgapp stream or any other message from a message stream
                # reads the message stream messages, as well as the MsgApp message from the msgapp stream
                msg_size = int("0x" + data[:24].replace(':',''), 16)
                data = data[24:] ; data_len -= 8 ; tcp_len -= 8 # removes message size (uint64)
                if (data[:2] == "08"):
                  msg_type = int("0x" + data[3:5], 16)
                if (data[6:8] == "10"):
                  msg_to = int("0x" + data[9:11], 16)
                if (data[12:14] == "18"):
                  msg_from = int("0x" + data[15:17], 16)
                if (data[18:20] == "20"):
                  if not(int("0x" + data[21:23], 16) & 0x80):
                    msg_term = int("0x" + data[21:23], 16)
                  else:
                    msg_term = (int("0x" + data[21:23], 16) & 0x7F) + (int("0x" + data[24:26], 16) << 7)
                if (data[78:80] == "50"):
                  msg_reject = int("0x" + data[81:83], 16)
                eprint("      size: (8+", msg_size, " ", hex(msg_size), "), type: ", msg_type, ", to: ", msg_to, ", from: ", msg_from, ", term: ", msg_term, ", reject: ", msg_reject, sep="")
                data = data[msg_size * 3:] ; data_len -= msg_size ; tcp_len -= msg_size # removes message
                # changes msg type from 8 to -1 , for LinkHeartbeats in a message stream
                if msg_type == 8 and msg_to == 0 and msg_from == 0: 
                  msg_type = -1

              if data[:5] == "0d:0a": # if follows \r\n
                data = data[6:] ; tcp_len -= 2 # removes \r\n. These two bytes are not included in the data_len
                eprint("      2 bytes '0d:0a' are removed", sep="")

              # edit to and from for LinkHeartbeats and MsgAppEntries
              if msg_type < 0:
                msg_term = msg_reject = 0
                for i in range(1, NODES+1):
                  for j in range(1, NODES+1):
                    if pkt.tcp.stream == stream["message"][str(i)+str(j)] or pkt.tcp.stream == stream["msgapp"][str(i)+str(j)]:
                      msg_to = i
                      msg_from = j
                
              ## In the following lines, the sample is created to be used by the estimator (we either use estimator, as now, or we create data to train it)
              # msgs.append([msg_type + (msg_reject / 2.0), msg_to, msg_from]) #, msg_term, msg_reject, pkt.sniff_timestamp, pkt.number])

              # if len(msgs) > SAMPLE_SIZE: # num of msgs will always be equal to SAMPLE_SIZE
              #   msgs.pop(0) # removes the oldest msg
              # if msg_type == 8 and msg_term > last_leader_term: # new leader, thus a failure happened. Estimator will tell us what kind of failure.
              #   last_leader_term = msg_term
              #   if len(msgs) == SAMPLE_SIZE:
              #     sample = [field for msg in msgs for field in msg]
              #     column_names = add_suffixes(SAMPLE_SIZE, ['msgType','sendTo','from'])
              #     df = pd.DataFrame(sample, column_names)
              #     df = df.T
              #     prediction = estimator.predict(df)
              #     bprint("Term: ", msg_term, " leader: ", msg_from, " failure: ", failure_labels[prediction[0]])
              #   else:
              #     bprint("Term: ", msg_term, " leader: ", msg_from, " failure: not enough msgs")

              ## In the following line, the data are collected to train the estimator (we either create data to train the estimator, as now, or we use it)
              oprint(msg_type, ",", msg_to, ",", msg_from, ",", msg_term, ",", msg_reject, ",", pkt.sniff_timestamp, ",", pkt.number)
          except Exception:
            eprint(pkt)
            eprint("(", len(data_print), ") ", data_print)
            raise

    #else:
    #  eprint(pkt)

  # before proceed to the next pcap file, we make sure that we won't read the same streams again
  for i in range(1, NODES+1):
    for j in range(1, NODES+1):
      if stream["message"][str(i)+str(j)] != -1:
        stream["message"][str(i)+str(j)] = -2
      if stream["msgapp"][str(i)+str(j)] != -1:
        stream["msgapp"][str(i)+str(j)] = -2
