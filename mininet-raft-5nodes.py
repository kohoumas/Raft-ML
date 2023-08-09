#!/usr/bin/python

#You may need to first execute: mn -c

import re
import string
from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mininet.link import Intf
from mininet.util import quietRun

import os
import sys
import signal
import time
from datetime import datetime
from random import randint
from subprocess import check_output

def check_process(name):
    output = []
    cmd = "ps -aef | grep -i '%s' | grep -v 'grep' | awk '{ print $2 }' > /tmp/out"
    os.system(cmd % name)
    with open('/tmp/out', 'r') as f:
        line = f.readline()
        while line:
            output.append(line.strip())
            line = f.readline()
            if line.strip():
                output.append(line.strip())
    return output

def add_flow(src, dst):
    os.system("ovs-ofctl add-flow s1 in_port=%d,eth_type=0x0800,ip_dst=192.168.1.%d,actions=output:%d" % (src, dst, dst)) 
    os.system("ovs-ofctl add-flow s1 in_port=%d,eth_type=0x0800,ip_dst=192.168.1.%d,actions=output:%d" % (dst, src, src)) 
    os.system("ovs-ofctl add-flow s1 in_port=%d,eth_type=0x0806,ip_dst=192.168.1.%d,actions=output:%d" % (src, dst, dst)) 
    os.system("ovs-ofctl add-flow s1 in_port=%d,eth_type=0x0806,ip_dst=192.168.1.%d,actions=output:%d" % (dst, src, src)) 
def add_flood_flow(src):
    os.system("ovs-ofctl add-flow s1 in_port=%d,eth_dst=FF:FF:FF:FF:FF:FF,actions=output:flood" % (src)) 
def del_flow(src, dst):
    os.system("ovs-ofctl del-flows s1 in_port=%d,eth_type=0x0800,ip_dst=192.168.1.%d" % (src, dst)) 
    os.system("ovs-ofctl del-flows s1 in_port=%d,eth_type=0x0800,ip_dst=192.168.1.%d" % (dst, src)) 
    os.system("ovs-ofctl del-flows s1 in_port=%d,eth_type=0x0806,ip_dst=192.168.1.%d" % (src, dst)) 
    os.system("ovs-ofctl del-flows s1 in_port=%d,eth_type=0x0806,ip_dst=192.168.1.%d" % (dst, src)) 


net = None
iterations = 10000
total_prediction_successes = 0

failure_num = {}
total_failure_num = 0
prediction_successes = {}

def myNet():

    CONTROLLER_IP='127.0.0.1'
    SLEEP_AFTER_TCPDUMP = 3
    SLEEP_AFTER_RAFT = 3
    SLEEP_AFTER_FAILURE = 5
    SLEEP_AFTER_RECOVERY = 5

    # Create network
    #net = Mininet( controller=RemoteController, link=TCLink, switch=OVSKernelSwitch)
    global net
    net = Mininet( controller=Controller, link=TCLink, switch=OVSKernelSwitch)

    # Create devices 
    ## Stations
    h1 = net.addHost( 'h1', ip="192.168.1.1/24")
    h2 = net.addHost( 'h2', ip="192.168.1.2/24")
    h3 = net.addHost( 'h3', ip="192.168.1.3/24")
    h4 = net.addHost( 'h4', ip="192.168.1.4/24")
    h5 = net.addHost( 'h5', ip="192.168.1.5/24")
    ## Switches
    s1 = net.addSwitch( 's1', dpid='00:00:00:00:00:00:00:01') #, failMode='standalone')

    # Create links 
    net.addLink(h1, s1) #, delay='40ms')
    net.addLink(h2, s1) #, delay='40ms')
    net.addLink(h3, s1) #, delay='40ms')
    net.addLink(h4, s1) #, delay='40ms')
    net.addLink(h5, s1) #, delay='40ms')
    
    # Create Controllers
    c1 = net.addController( 'c1', controller=RemoteController, ip=CONTROLLER_IP, port=6633)

    net.start()

    # Start Controllers
    c1.start()
    s1.start( [c1] )

    predict = False # Live prediction is done

    os.system("cd ~/raftexample; rm -rf raftexample-*; rm -f raft-output-*.txt; rm -f predicted-data.txt")

    # Start tcpdumps
    print("\n\n-- Tcpdump or pyshark starts at nodes")
    os.system("cd ~/raftexample; tcpdump -i s1-eth1 -w raft-output-1.pcap &")
    os.system("cd ~/raftexample; tcpdump -i s1-eth2 -w raft-output-2.pcap &")
    os.system("cd ~/raftexample; tcpdump -i s1-eth3 -w raft-output-3.pcap &")
    os.system("cd ~/raftexample; tcpdump -i s1-eth4 -w raft-output-4.pcap &")
    os.system("cd ~/raftexample; tcpdump -i s1-eth5 -w raft-output-5.pcap &")
    if predict:
      os.system("cd ~/raftexample; python3 pyshark_read_raft.py 2>/tmp/pyshark-error.txt 1>predicted-data.txt &")
    time.sleep(SLEEP_AFTER_TCPDUMP)

    # Setup Openflow entries and initialize statistics structures
    print("-- Openflow entries and statistic structures are initiated")
    for src in range(1,6):
        add_flood_flow(src)
        failure_num["%s" % src] = 0
        for dst in range(src+1,6):
            add_flow(src, dst)
            failure_num["%s-%s" % (src, dst)] = 0
            failure_num["%s-%s" % (dst, src)] = 0
    prediction_successes = failure_num.copy()

    # Start Raft
    print("-- Raftexample starts at nodes")
    h1.cmd("cd ~/raftexample; goreman -f Procfile-5-1 start > raft-output-1.txt &")
    h2.cmd("cd ~/raftexample; goreman -f Procfile-5-2 start > raft-output-2.txt &")
    h3.cmd("cd ~/raftexample; goreman -f Procfile-5-3 start > raft-output-3.txt &")
    h4.cmd("cd ~/raftexample; goreman -f Procfile-5-4 start > raft-output-4.txt &")
    h5.cmd("cd ~/raftexample; goreman -f Procfile-5-5 start > raft-output-5.txt &")
    time.sleep(SLEEP_AFTER_RAFT)

    print("-- Experiment starts")

    nodes = [1, 2, 3, 4, 5]
    numline = [0] * len(nodes) 
    termfile = [0] * len(nodes)
    leaderfile = [-1] * len(nodes)
    leaderfile[0] = -2 # in order all not to be equal at the start

    predictline = 1 # ignore first line of predict-data. It is the prediction for the failure happened before the first election, which does not happen.
    term_predict = 0
    leader_predict = 0

    start = datetime.now()

    for iteration in range(iterations):

      ## Read who is the leader, from raft-output files
      repeat = True
      while repeat:
        for n in nodes:
          with open("/root/raftexample/raft-output-%d.txt" % n, "r") as fp:
            checkedlines = numline[n-1]
            while checkedlines!=0:
              line = fp.readline()
              if ("raft.node" in line) and ("leader" in line):
                checkedlines -= 1
            line = fp.readline()
            while line:
              if ("raft.node" in line) and ("leader" in line) and re.match("\d\d:\d\d:\d\d", line[5:]):
                #print("  %s" % (line))
                numline[n-1]+=1
                words = line.split()
                change = words[8]
                if change == "elected":
                  termfile[n-1] = int(words[-1])
                  leaderfile[n-1] = int(words[10])
                elif change == "changed":
                  termfile[n-1] = int(words[-1])
                  leaderfile[n-1] = int(words[13])
              line = fp.readline()
        repeat = False
        for n in nodes[:-1]:
          if leaderfile[n-1] != leaderfile[n] or termfile[n-1] != termfile[n]:
            repeat = True
            break
      leader = leaderfile[0]
      term = termfile[0]
      print("Iter: %d, @nodes %s => leader: %d, term: %d" % (iteration, nodes, leader, term))

      if predict and iteration and (term_predict != term or leader_predict != leader):
        print("Sync error between all raft-output-*.txt (leader: %d, term: %d) and predict-data.txt (leader: %d, term: %d)" % (leader, term, leader_predict, term_predict))
        exit_gracefully() 

      ## Create a random failure
      failure_type = randint(0, 8) # failure_type == 0 => node failure, failure_type > 0 => link failure.
      failure_label = None
      if failure_type > 4:
        other = (leader - 1 + failure_type - 4) % 5 + 1
        
        print("  Link %d-%d goes down for %d sec" % (leader, other, SLEEP_AFTER_FAILURE))
        del_flow(leader, other) 
        time.sleep(SLEEP_AFTER_FAILURE)
        print("  Link %d-%d goes up and wait for %d sec before next iteration" % (leader, other, SLEEP_AFTER_RECOVERY))
        add_flow(leader, other)
        failure_label = "%d-%d" % (leader, other)
      else:
        print("  Node %d goes down for %d sec" % (leader, SLEEP_AFTER_FAILURE))
        if leader == 1:
          h1.cmd("goreman run stop-all")
        elif leader == 2:
          h2.cmd("goreman run stop-all")
        elif leader == 3:
          h3.cmd("goreman run stop-all")
        elif leader == 4:
          h4.cmd("goreman run stop-all")
        elif leader == 5:
          h5.cmd("goreman run stop-all")
        time.sleep(SLEEP_AFTER_FAILURE)
        numline[leader-1] = 0
        termfile[leader-1] = 0
        leaderfile[leader-1] = -2
        print("  Node %d goes up and wait for %d sec before next iteration" % (leader, SLEEP_AFTER_RECOVERY))
        if leader == 1:
          h1.cmd("cd ~/raftexample; goreman -f Procfile-5-1 start >> raft-output-1.txt &")
        elif leader == 2:
          h2.cmd("cd ~/raftexample; goreman -f Procfile-5-2 start >> raft-output-2.txt &")
        elif leader == 3:
          h3.cmd("cd ~/raftexample; goreman -f Procfile-5-3 start >> raft-output-3.txt &")
        elif leader == 4:
          h4.cmd("cd ~/raftexample; goreman -f Procfile-5-4 start >> raft-output-4.txt &")
        elif leader == 5:
          h5.cmd("cd ~/raftexample; goreman -f Procfile-5-5 start >> raft-output-5.txt &")
        failure_label = str(leader)
      global total_failure_num
      total_failure_num += 1
      global failure_num
      failure_num[failure_label] += 1
      time.sleep(SLEEP_AFTER_RECOVERY)

      ## Read from predicted-data what is the prediction of ML classifier, and write it.
      if predict:
        with open("/root/raftexample/predicted-data.txt", "r") as fp:
          checkedlines = predictline
          while checkedlines!=0:
            line = fp.readline()
            if "failure:" in line:
              checkedlines -= 1
          line = fp.readline()
          while line:
            if "failure:" in line:
              predictline += 1
              words = line.split()
              term_predict = int(words[1])
              leader_predict = int(words[3])
              failure_predict = string.join(words[5:], sep=" ")
              if failure_predict == failure_label:
                print("  Correct prediction of %s" % failure_label)
                global total_prediction_successes
                total_prediction_successes += 1
                global prediction_successes
                prediction_successes[failure_label] += 1
              else:
                print("  Wrong   prediction of %s as %s" % (failure_label, failure_predict))
              break
            line = fp.readline()

      #h1.cmd("curl -L http://127.0.0.1:12380/my-key1 -XPUT -d " + str(i))
      #time.sleep(3)
      #h2.cmd("curl -L http://127.0.0.1:22380/my-key2 -XPUT -d " + str(i))
      #time.sleep(3)
      #h3.cmd("curl -L http://127.0.0.1:32380/my-key3 -XPUT -d " + str(i))
      #time.sleep(3)

    end = datetime.now()
    print (end - start)

    #CLI( net )
    exit_gracefully()

def exit_gracefully():
    global total_prediction_successes, total_failure_num
    success_rate = 100.0 * total_prediction_successes/total_failure_num if total_failure_num else -1.0
    print("-- Prediction success rate: %.1f" % (success_rate))
    for key in sorted(prediction_successes):
      failure_rate = 100.0 * failure_num[key]/total_failure_num if total_failure_num else -1.0
      success_rate = 100.0 * prediction_successes[key]/failure_num[key] if failure_num[key] else -1.0
      print("--   Prediction success rate for failure %s (freq: %.1f): %.1f" % (key, failure_rate, success_rate))
    print("-- Experiment stops")
    # kill pyshark_raft_read.py process
    for pid in check_process("pyshark_read_raft.py"):
      os.system("kill -9 " + pid)
    global net
    net.stop()
    sys.exit(0)

def kill_signal_handler(signal, frame):
    exit_gracefully()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, kill_signal_handler)
    setLogLevel( 'info' )
    myNet()
