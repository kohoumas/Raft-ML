#!/usr/bin/python3

import sys
import pandas as pd

from functions import *

if len(sys.argv) < 4:
	print("Give arguments as below:\n1)max_heartbeats\n2)Sample Size\n3)Dataset Filepath (txt format)")
	exit(-1)

max_heartbeats = int(sys.argv[1])
history = int(sys.argv[2])

filepath = sys.argv[3]

df = pd.read_csv(filepath, header=None)
df.columns = ["msgType","sendTo","from","term","voteResp","relativeTime","tcpPacket"]

if df.empty:
	print("There is no such input file! Exiting...\n")
	exit(-1)

start = filepath.find("data-output-") + len("data-output-")
end = filepath.find(".txt")
node = filepath[start:end]

relativeTime = relative_time_converter(df) # all relative times change in order the one of the first packet to be zeroed
df_removed_relativeTime = df.drop(columns=['relativeTime'])
df_only_relativeTime = pd.DataFrame(relativeTime, columns=["relativeTime"])
df = df_removed_relativeTime.join(df_only_relativeTime)

run_all(df=df, history=history, max_heartbeats=max_heartbeats, title_prefix="dataset_output_"+node)
