#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric

#NODES = 3
NODES = 5
ROLES = 4 ## 1 : OBSERVER, 2 : PREVIOUS LEADER, 3 : NEXT LEADER, 4 : OTHERS

column_names = ["msgType", "sendTo", "from", "term", "relativeTerm", "voteResp", "time", "relativeTime"]
column_types = {"msgType": float, "sendTo": int, "from": int, "term": int, "relativeTerm": int, "voteResp": int, "time": float, "relativeTime": float}
failure_labels = ["1", "2", "3", "1-2", "1-3", "2-1", "2-3", "3-1", "3-2"]
if NODES == 5 :
	failure_labels = ["1", "2", "3", "4", "5", 
		"1-2", "1-3", "1-4", "1-5", 
		"2-1", "2-3", "2-4", "2-5", 
		"3-1", "3-2", "3-4", "3-5",
		"4-1", "4-2", "4-3", "4-5",
		"5-1", "5-2", "5-3", "5-4"]


def relative_time_converter(df):
	firstTime = df.loc[0, 'relativeTime']
	relativeTimes = [0.0]
	for row in df[1:].itertuples():
		time = row.relativeTime
		relativeTime = time - firstTime
		relativeTimes.append(relativeTime)
	return relativeTimes


def read_metadata_file(metadata_file, max_heartbeats):
	failures = []
	terms = []
	file = open(metadata_file, "r")
	for line in file:
		if len(terms) >= max_heartbeats:
			break
		elif "leader: " in line and not("Iter: 0" in line):
			words = line.split()
			term = int(words[-1])
			terms.append(term)
		elif "Node" in line and "goes down" in line:
			words = line.split()
			node = words[1]
			failures.append(failure_labels.index(node))
		elif "Link" in line and "goes down" in line:
			words = line.split()
			link = words[1]
			failures.append(failure_labels.index(link))
		# for backwards compatibility with previous experiments
		elif "s1-eth2 goes down" in line:
			failures.append(failure_labels.index("1-2"))
		elif "s1-eth3 goes down" in line:
			failures.append(failure_labels.index("1-3"))
		elif "s2-eth2 goes down" in line:
			failures.append(failure_labels.index("2-3"))
		elif "s2-eth3 goes down" in line:
			failures.append(failure_labels.index("2-1"))
		elif "s3-eth2 goes down" in line:
			failures.append(failure_labels.index("3-1"))
		elif "s3-eth3 goes down" in line:
			failures.append(failure_labels.index("3-2"))
	file.close()
	if (len(terms) != len(failures)) : 
		print("  Num of terms: %d != Num of failures: %d" % (len(terms), len(failures)))
		exit(-1)
	return failures, terms


def find_heartbeats(df, terms, max_heartbeats):
	tuples = []
	term_index = 0
	for row in df.itertuples(): 
		row_index = row[0]
		msgType = int(row.msgType)
		term = int(row.term)
		if len(tuples) == max_heartbeats:
			break
		# add heartbeats that don't have the same term with an already added heartbeat
		if (msgType == 8) and (term in terms[term_index:]):
			term_index = terms.index(term) + 1
			tuples.append([row_index, term])
	return tuples
	

def create_packet_list(df, starting_row, ending_row):
	sub_packet_list = []
	row = df.loc[ending_row]
	term_ref = int(row.term)
	time_ref = float(row.relativeTime)
	for row_index in range(starting_row, ending_row + 1):
		row = df.loc[row_index]
		msgType = float(row.msgType)
		sendTo = int(row.sendTo)
		recvFrom = int(row['from'])
		term = int(row.term)
		voteResp = int(row.voteResp)
		time = float(row.relativeTime)

		#if voteResp == 1 and msgType != 6.0:
		#	print("Error: voteResp = %d and msgType = %.1f" % (voteResp, msgType))
		assert(not(voteResp == 1 and (msgType != 6.0 and msgType != 3.0)))
		msgType += (voteResp / 2.0) # voteResp affects only the vote msgs: when voteResp is 1, the mesgType changes from 6 to 6.5

		relativeTerm = int(term_ref - term) if term else 0
		relativeTime = time_ref - time

		sub_packet_list.append([msgType, sendTo, recvFrom, term, relativeTerm, voteResp, time, relativeTime])
	return sub_packet_list


def packet_to_df(packet_list):
	df_list = []
	for packet in packet_list:
		df = pd.DataFrame(packet, column_names)
		df = df.T
		df = df.astype(column_types)
		df_list.append(df)
	return df_list


def create_samples(df, failure_terms, history, max_heartbeats):
	tuples = find_heartbeats(df, failure_terms, max_heartbeats)
	print("done find_heartbeats... Num of heartbeats: %d" % len(tuples))
	tuple_terms = failure_terms
	if len(failure_terms) > len(tuples):
		print("  Num of failures %d > Num of heartbeats %d" % (len(failure_terms), len(tuples)))
		print("  Missing heartbeats for failures at terms: ", end="")
		tuple_terms = [tupl[1] for tupl in tuples]
		for t in [term for term in failure_terms if term not in tuple_terms]:
			print("%d, " % t, end="")
		print("")

	packet_list = []
	for row in tuples:
		sample_packet_list = create_packet_list(df, row[0]-history+1, row[0])
		packet_list += sample_packet_list
	print("done create_packet_list... Num of used packets: %d" % len(packet_list))

	df_list = packet_to_df(packet_list)
	print("done packet_to_df...")

	samples = []
	suffix = 0
	for i in range(len(df_list)):
		if suffix == 0:
			suffix = history - 1
			samples.append(df_list[i].add_suffix(str(suffix)))
		else:
			suffix -= 1
			samples[-1] = samples[-1].join(df_list[i].add_suffix(str(suffix)))

	return samples, tuple_terms


def to_dataframe(samples, failures, history):
	new_df = pd.DataFrame()
	for sample in samples:
		heartbeatFrom = 0
		heartbeatIndex = history
		voteReqFrom = 0
		voteReqIndex = history
		for i in range(history-1, 0, -1):
			if (int(sample["msgType"+str(i)]) == 8):
				heartbeatFrom = int(sample["from"+str(i)])
				heartbeatIndex = i
			elif (int(sample["msgType"+str(i)]) == 5) and (voteReqIndex == history):
				voteReqFrom = int(sample["from"+str(i)])
				voteReqIndex = i
		sample_extra = pd.DataFrame([heartbeatFrom, heartbeatIndex, voteReqFrom, voteReqIndex], 
			                        ["LastHeartbeatFrom", "LastHeartbeatIndex", "FirstVoteReqFrom", "FirstVoteReqIndex"])
		sample_extra = sample_extra.T
		sample_extra = sample_extra.astype({"LastHeartbeatFrom": int, 
			                                "LastHeartbeatIndex": int, 
			                                "FirstVoteReqFrom": int, 
			                                "FirstVoteReqIndex": int})
		sample = sample.join(sample_extra)
		new_df = new_df.append(sample, ignore_index=False, sort=False)
	new_df["failure"] = np.array(failures)
	return new_df


def run_all(df, history, max_heartbeats, title_prefix):
	failures, failure_terms = read_metadata_file("meta-data.txt", max_heartbeats)
	print("Num of failures in metadata: %d" % len(failures))

	samples, sample_terms = create_samples(df, failure_terms, history, max_heartbeats)
	print("Num of samples in dataframe: %d" % len(samples))

	if len(failures) > len(samples):
		used_failures = []
		for i in range(len(failure_terms)):
			if failure_terms[i] in sample_terms:
				used_failures.append(failures[i])
		failures = used_failures
		print("Num of used failures: %d" % len(used_failures))

	new_df = to_dataframe(samples, failures, history)
	title = title_prefix + "_size" + str(len(samples)) + "_history" + str(history) + ".csv"
	new_df.to_csv(title)
	print("Creation of csv is completed!")


def add_suffixes(history, column_order):
	output_cols = []
	for i in range(history-1, -1, -1):
		for j in range(len(column_order)):
			output_cols.append(column_order[j]+str(i))
	return(output_cols)


# def hamming_kernel(x, y):
# 	# x,y of the same length
# 	dist = DistanceMetric.get_metric('hamming')
# 	return 1.0 - dist.pairwise(x,y)


def enhance_failure_labels(df):
	# change failure labels from e.g. 1 to 1>2, 1>3, etc.
	failure_labels = []
	df.failure = df.failure + NODES*(NODES-1)
	index = 0
	for i in range(1, NODES+1):
		for j in range(1, NODES+1):
			if j != i:
				failure_labels.append("%d>%d" % (i,j))
				df.loc[(df.failure == (NODES*(NODES-1) + (i-1))) & (df.from0 == j), 'failure'] = index
				index += 1
	for i in range(1, NODES+1):
		for j in range(1, NODES+1):
			if j != i:
				failure_labels.append("%d-%d" % (i,j))
	df.loc[(df.failure >= NODES*NODES), 'failure'] = df.loc[(df.failure >= NODES*NODES), 'failure'] - NODES
	return failure_labels, df

## The following is working with only 6 labels, independently of the # of NODES (use dtree_play).
def enhance_failure_labels_play_old(df):
	failure_labels = ["1>3", "2>1", "2>3", "1-3", "2-1", "2-3"]
	df.loc[(df.failure == 0), 'failure'] = 100 ## temporary label 100 for 1>3
	df.loc[(df.failure > 0) & (df.failure < NODES) & (df.from0 == 1), 'failure'] = 101 ## temporary label 101 for 2>1
	df.loc[(df.failure > 0) & (df.failure < NODES) & (df.from0 != 1), 'failure'] = 102 ## temporary label 102 for 2>3
	df.loc[(df.failure >= NODES) & (df.failure < 2*NODES-1), 'failure'] = 103 ## temporary label 103 for 1-3
	for i in range(1, NODES):
		df.loc[(df.failure == NODES + i*(NODES-1)), 'failure'] = 104 ## temporary label 104 for 2-1
		df.loc[(df.failure > NODES + i*(NODES-1)) & (df.failure < NODES + (i+1)*(NODES-1)), 'failure'] = 105 ## temporary label 105 for 2-3
	df.failure = df.failure - 100
	return failure_labels, df

## The following is working with only 2 labels, link or node failure.
def enhance_failure_labels_play(df):
	failure_labels = [">", "-"]
	df.loc[(df.failure == 0), 'failure'] = 100 ## temporary label 100 for 1>3
	df.loc[(df.failure > 0) & (df.failure < NODES) & (df.from0 == 1), 'failure'] = 100 ## temporary label 101 for 2>1
	df.loc[(df.failure > 0) & (df.failure < NODES) & (df.from0 != 1), 'failure'] = 100 ## temporary label 102 for 2>3
	df.loc[(df.failure >= NODES) & (df.failure < 2*NODES-1), 'failure'] = 101 ## temporary label 103 for 1-3
	for i in range(1, NODES):
		df.loc[(df.failure == NODES + i*(NODES-1)), 'failure'] = 101 ## temporary label 104 for 2-1
		df.loc[(df.failure > NODES + i*(NODES-1)) & (df.failure < NODES + (i+1)*(NODES-1)), 'failure'] = 101 ## temporary label 105 for 2-3
	df.failure = df.failure - 100
	return failure_labels, df


MSG_VALUES = 3 ## How many message values (msgType, sendTo, from) are transformed to msg-id
MSG_GROUPS = 6 ## LinkHeartbeats, Heartbeat Reqs, Heartbeat Reps, 
               ##   Vote/Append Reqs, Vote/Append Pos/ve Reps, Vote/Append Neg/ve Reps.
MSG_BITS = MSG_GROUPS * 2 * (ROLES - 1) ## How many binary bits will be used for each msg-id, 
                                        ##   when it is binary expressed. 
                                        ##   MSG_GROUPS x [(ROLES - 1 = # msg senders, apart from 1) +
                                        ##                 (ROLES - 1 = # msg receivers, when node 1 is the sender)]

def transform_sample(sample, history, binary=True): 
	## Each of the first tuples (msgType, sendTo, from), of total number equal to history, 
	##   is transformed to either a msg id or a bit sequence of length equal to MSG_BITS.
	transformed_sample = []
	processed_sample = sample.copy() ## This block changes each from_id of the sample, 
	                                 ##   to one of the MSG_ROLES
	first_heartbeat_from = int(sample[-2]) ## LastHeartbeatFrom
	first_heartbeat_index = int(sample[-1]) ## LastHeartbeatIndex
	if first_heartbeat_index >= history:  ## Useful in cases that we reduce history of samples.
		first_heartbeat_from = 0
	last_heartbeat_from = sample[MSG_VALUES*(history-1)+2] ## from0 
	msg_freq_role = [0] * (ROLES-1) # number of appearances of msgs with sender or receiver specific role (PREV. LEADER, NEW LEADER or OTHER).
	msg_last_role = [history] * (ROLES-1) # position of the most recent appearance of a msg sent to or received by a specific role (PREV. LEADER, NEW LEADER or OTHER). Position is history if there is no msg from this role.
	msg_first_role = [history] * (ROLES-1) # position of the most past appearance of a msg sent to or received by a specific role (PREV. LEADER, NEW LEADER or OTHER). Position is history if there is no msg from this role.
	for i in range(history):
		role_id = 0  # the role that is either sender or receiver of the message, apart from role OBSERVER
		sendTo_id = sample[MSG_VALUES*i+1] 
		if sendTo_id == 1:
			role_id = 1
		elif sendTo_id == first_heartbeat_from:
			role_id = 2
		elif sendTo_id == last_heartbeat_from:
			role_id = 3
		else:
			role_id = 4
		processed_sample[MSG_VALUES*i+1] = role_id
		if (role_id > 1):
			msg_freq_role[role_id - 2] += 1
			msg_last_role[role_id - 2] = (history - i)
			if msg_first_role[role_id - 2] == history:
				msg_first_role[role_id - 2] = (history - i)
					
		from_id = sample[MSG_VALUES*i+2] 
		if from_id == 1:
			role_id = 1
		elif from_id == first_heartbeat_from:
			role_id = 2
		elif from_id == last_heartbeat_from:
			role_id = 3
		else:
			role_id = 4
		processed_sample[MSG_VALUES*i+2] = role_id
		if (role_id > 1):
			msg_freq_role[role_id - 2] += 1
			msg_last_role[role_id - 2] = (history - i)
			if msg_first_role[role_id - 2] == history:
				msg_first_role[role_id - 2] = (history - i)

	if first_heartbeat_from != 0 and first_heartbeat_from != 1:
		processed_sample[-2] = 2
	else:
		processed_sample[-2] = 0
	sample = processed_sample
	
	msg_freq = [0] * MSG_BITS # number of appearances of each msg id.
	msg8_freq = 0 # number of appearances of heartbeats
	msg_last = [history] * MSG_BITS # position of the most recent appearance of each msg id. It is history if there is no msg of this id.
	msg_first = [history] * MSG_BITS # position of the most past appearance of each msg id. It is history if there is no msg of this id.
	for i in range(history): ## The first (MSG_VALUES x history) sample integers are read in this loop, 
                                 ##   and transformed to (MSG_BITS x history) bits.
		msg_id = 0
		group_id = None
		type_id = sample[MSG_VALUES*i]
		sendTo_id = sample[MSG_VALUES*i+1] 
		from_id = sample[MSG_VALUES*i+2] 
		if type_id == -2.0 or type_id == -1.0: # link-heartbeats through msgApp and message stream respectively
			group_id = 0
			#if from_id == 1: sendTo_id = 4 ## In case of link-heartbeats from node 1, we don't care who is their receiver or sender. 
			#elif sendTo_id == 1: from_id = 4
		elif type_id == 8.0: # heartbeat requests
			group_id = 1
			msg8_freq += 1
			#if from_id == 1: sendTo_id = 4 ## In case of heartbeat msgs from node 1, we don't care who is their receiver. 
                        #                               ## It helps if it is the same for all heartbeats, so we use arbitrarily the 4th role.
		elif type_id == 9.0: # heartbeat replies
			group_id = 2
		elif type_id == -3.0 or type_id == 3.0 or type_id == 3.5 or type_id == 5.0: # append requests (msgApp stream), append positive requests (message stream), append negative requests (message stream) , vote requests
			group_id = 3
		elif type_id == 4.0 or type_id == 6.0: # append replies, vote positive replies
			group_id = 4
		elif type_id == 6.5: # vote negative replies
			group_id = 5
		assert(group_id != None)
		assert(group_id == 1 or i != (history-1)) # last message is always heartbeat request.
		msg_id += group_id * 2 * (ROLES - 1)
		if from_id == 1:
			msg_id += (sendTo_id - 2)
		else:
			msg_id += (ROLES - 1) + (from_id - 2)
		msg_id = int(msg_id)
		msg_freq[msg_id] += 1
		msg_last[msg_id] = (history - i)
		if msg_first[msg_id] == history:
			msg_first[msg_id] = (history - i) 
		if binary:
			msg = [0] * MSG_BITS; 
			if i == (history-1):
				msg[msg_id] = 1 #history # last message is weighted (history) times more than others, since it is the indicator of the next leader
			else:
				msg[msg_id] = 1
			transformed_sample += msg
		else:
			transformed_sample.append(msg_id)

	## The remaining history x relative times are copied unchanged.
	#for i in range(MSG_VALUES*history, len(sample) - 2): # Last sample integers, with index higher than (MSG_BITS x history), APART FROM the very last two ones, remain unchanged.
	#	transformed_sample.append(sample[i])

	## The before-last value is the previous leader id, which is copied either as binary or as it is.
	prev_leader_id = int(sample[-2]) # LastHeartbeatFrom, after replacing node ids with roles, is equal to either 0, 1 or 2 (=UNKNOWN, OBSERVER or PREVIOUS LEADER).
	if binary: 
		prev_leader_bits = [0, 0]
		if prev_leader_id > 0: # 1 or 2 (=OBSERVER or PREVIOUS LEADER).
			prev_leader_bits[prev_leader_id - 1] = 1 # history # last message is weighted 20 times more than others, since it is the indicator of the previous leader
			                                         # [1, 0], [0, 1] or [0, 0] for OBSERVER, PREVIOUS LEADER or unknown prev. leader respectively.
		#transformed_sample += prev_leader_bits
	else:
		transformed_sample.append(prev_leader_id)

	## Two new values are added at the end, the sum and variance of the relative times.
	#sum_rel_time = 0.0 ## the sum of the relative times
	#for i in range(MSG_VALUES*history+1, len(sample) - 2, 1):
	#	sum_rel_time += sample[i]
	#transformed_sample.append(sum_rel_time) 
	#var_rel_time = 0.0 ## the var of the relative times
	#for i in range(MSG_VALUES*history+1, len(sample) - 3, 1):
	#	var_rel_time += (sample[i] - sample[i+1])**2
	#transformed_sample.append(var_rel_time)
	## Three new sequences are added at the end, the numbers and the most recent/past positions for each msg id.
	#transformed_sample += msg_freq
	#transformed_sample += msg_last
	#transformed_sample += msg_first
	## Three new sequences are added at the end, the numbers and the most recent/past positions of the msgs sent to or received by either role 2, 3 or 4.
	#transformed_sample += msg_freq_role
	#transformed_sample += msg_last_role 
	#transformed_sample += msg_first_role 
	## Number of heartbeats, sent or received, is added at the end.
	#transformed_sample.append(msg8_freq)
	return transformed_sample

def inverse_transform_sample(transformed_sample, history): 
	sample = []
	for i in range(history):
		msg_type = 0; msg_sendTo = 0; msg_from = 0
		index = np.argmax(transformed_sample[(MSG_BITS*i):(MSG_BITS*i+MSG_BITS)])
		if   index ==  0:  msg_type = -1; msg_sendTo = 2; msg_from = 1;
		elif index ==  1:  msg_type = -1; msg_sendTo = 3; msg_from = 1;
		elif index ==  2:  msg_type = -1; msg_sendTo = 4; msg_from = 1;
		elif index ==  3:  msg_type = -1; msg_sendTo = 1; msg_from = 2;
		elif index ==  4:  msg_type = -1; msg_sendTo = 1; msg_from = 3;
		elif index ==  5:  msg_type = -1; msg_sendTo = 1; msg_from = 4;
		elif index ==  6:  msg_type =  8; msg_sendTo = 2; msg_from = 1;
		elif index ==  7:  msg_type =  8; msg_sendTo = 3; msg_from = 1;
		elif index ==  8:  msg_type =  8; msg_sendTo = 4; msg_from = 1;
		elif index ==  9:  msg_type =  8; msg_sendTo = 1; msg_from = 2;
		elif index == 10:  msg_type =  8; msg_sendTo = 1; msg_from = 3;
		elif index == 11:  msg_type =  8; msg_sendTo = 1; msg_from = 4;
		elif index == 12:  msg_type =  9; msg_sendTo = 2; msg_from = 1;
		elif index == 13:  msg_type =  9; msg_sendTo = 3; msg_from = 1;
		elif index == 14:  msg_type =  9; msg_sendTo = 4; msg_from = 1;
		elif index == 15:  msg_type =  9; msg_sendTo = 1; msg_from = 2;
		elif index == 16:  msg_type =  9; msg_sendTo = 1; msg_from = 3;
		elif index == 17:  msg_type =  9; msg_sendTo = 1; msg_from = 4;
		elif index == 18:  msg_type =  5; msg_sendTo = 2; msg_from = 1;
		elif index == 19:  msg_type =  5; msg_sendTo = 3; msg_from = 1;
		elif index == 20:  msg_type =  5; msg_sendTo = 4; msg_from = 1;
		elif index == 21:  msg_type =  5; msg_sendTo = 1; msg_from = 2;
		elif index == 22:  msg_type =  5; msg_sendTo = 1; msg_from = 3;
		elif index == 23:  msg_type =  5; msg_sendTo = 1; msg_from = 4;
		elif index == 24:  msg_type =  6; msg_sendTo = 2; msg_from = 1;
		elif index == 25:  msg_type =  6; msg_sendTo = 3; msg_from = 1;
		elif index == 26:  msg_type =  6; msg_sendTo = 4; msg_from = 1;
		elif index == 27:  msg_type =  6; msg_sendTo = 1; msg_from = 2;
		elif index == 28:  msg_type =  6; msg_sendTo = 1; msg_from = 3;
		elif index == 29:  msg_type =  6; msg_sendTo = 1; msg_from = 4;
		elif index == 30:  msg_type =  7; msg_sendTo = 2; msg_from = 1;
		elif index == 31:  msg_type =  7; msg_sendTo = 3; msg_from = 1;
		elif index == 32:  msg_type =  7; msg_sendTo = 4; msg_from = 1;
		elif index == 33:  msg_type =  7; msg_sendTo = 1; msg_from = 2;
		elif index == 34:  msg_type =  7; msg_sendTo = 1; msg_from = 3;
		elif index == 35:  msg_type =  7; msg_sendTo = 1; msg_from = 4;
		sample += [msg_type, msg_sendTo, msg_from]
	for i in range(MSG_BITS*history, len(transformed_sample)):
		sample.append(transformed_sample[i])
	return sample

def print_sample(x, history):
	for i in range(history):
		#print("%2+d:%2d<%d@%.2f, " % (history-i-1, x[MSG_VALUES*i], x[MSG_VALUES*i+2], x[MSG_VALUES*history+2*i+1]), end='')
		#print("%2d:%+d %d<%d@%.1f, " % (history-i-1, x[MSG_VALUES*i], x[MSG_VALUES*i+1], x[MSG_VALUES*i+2], x[MSG_VALUES*history+i+1]), end='')
		if True: #x[MSG_VALUES*i] == -3 or x[MSG_VALUES*i] == 3:
			print("%+d %d<%d, " % (x[MSG_VALUES*i], x[MSG_VALUES*i+1], x[MSG_VALUES*i+2]), end='')		
		else:
			print("    < , ", end='')
	print()
