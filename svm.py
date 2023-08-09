#!/usr/bin/python3

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from joblib import dump, load

from functions import *

if len(sys.argv)<2:
    print("Dataset filename(s) is/are not given!\n\n")
    exit(-1)

#history = int(sys.argv[1], 10)

datasets = []
for i in range(1,len(sys.argv)):
	dataset_filename = sys.argv[i]
	datasets.append(pd.read_csv(dataset_filename))
dataset = pd.concat(datasets)

start = dataset_filename.find("history") + len("history")
end = dataset_filename.find(".csv")
history = dataset_filename[start:end]
history = int(history, 10)

columns = add_suffixes(history, ['msgType','sendTo','from'])
columns += add_suffixes(history, ['relativeTime'])
columns += ["LastHeartbeatFrom", "LastHeartbeatIndex"]

failure_labels, dataset = enhance_failure_labels_play(dataset)

# filter same labels
filter_label_indexes = range(2) # range(6) # all indexes for the 'play' version
#filter_label_indexes = [index for index in range(len(failure_labels)) if index not in filter_label_indexes] # the complement set of indexes
filter_labels = [failure_labels[index] for index in filter_label_indexes]
dataset = dataset[dataset.failure.isin(filter_label_indexes)]
for index, label in enumerate(filter_labels):
    dataset.loc[(dataset.failure == failure_labels.index(label)), 'failure'] = index
failure_labels = filter_labels

data = dataset[columns]
data = data.values
transformed_data = [transform_sample(sample, history) for sample in data]
target = dataset['failure']
target = target.values

x_train, x_test, y_train, y_test = train_test_split(transformed_data, target, test_size=0.1, shuffle=True)

estimator = svm.SVC(kernel='linear', C=1, probability=True) 
#estimator = svm.SVC(kernel=hamming_kernel, C=1)
estimator.fit(x_train, y_train)
#dump(estimator, 'svm.joblib') # saves estimator to a file

#print(estimator.support_)

# Testing
y_pred = estimator.predict(x_train)
score = accuracy_score(y_train, y_pred)
score = int(100*score)
print("score on training (%d samples) = %d%%" % (len(y_train), score))
y_pred = estimator.predict(x_test)
score = accuracy_score(y_test, y_pred)
score = int(100*score)
print("score on testing (%d samples) = %d%%" % (len(y_test), score))

# real_data = data.tolist()
# support_vectors = [transformed_data[index]+[target[index]] for index in estimator.support_]
# real_vectors = [real_data[index]+[target[index]] for index in estimator.support_]
# dataset_last_sample = [0]
# for dataset in datasets:
# 	dataset_last_sample.append(dataset_last_sample[-1] + len(dataset))
# dataset_last_sample.pop(0)

# print("Num of vectors: %d\n" % len(support_vectors))

# for f in range(len(failure_labels)):
# 	print("Failure: %s" % (failure_labels[f]))
# 	support_vectors_failure = [vector for vector in support_vectors if vector[-1]==f]
# 	real_vectors_failure = [vector for vector in real_vectors if vector[-1]==f]
# 	print("Num of vectors: %d == %d" % (len(support_vectors_failure), estimator.n_support_[f]))
# 	assert(len(support_vectors_failure) == estimator.n_support_[f])
# 	for vector_num in range(len(support_vectors_failure)):
# 		vector_index = estimator.support_[vector_num]
# 		print("%04d. sample %04d from " % (vector_num, vector_index), end='')
# 		for dataset_num, last_sample_index in enumerate(dataset_last_sample):
# 			if vector_index <= last_sample_index:
# 				print("%dth set: " % (dataset_num), end='')
# 				break
# 		#print(support_vectors_failure[i])
# 		print_sample(real_vectors_failure[vector_num], history)
# 	print()

# index = 0 
# for x, y in zip(data, target):
# 	index += 1
# 	transformed_x = transform_sample(x, history)
# 	y_predict = estimator.predict([transformed_x])[0]
# 	probs = estimator.predict_proba([transformed_x])[0]
# 	if y != y_predict:
# 		print("%3d) " % index, end='')
# 		print("%3s predicted as %3s : probs %2d%% < %2d%% " % (failure_labels[y], failure_labels[y_predict], 100*probs[y], 100*probs[y_predict]))

errors = [[0 for i in range(len(failure_labels))] for j in range(len(failure_labels))]
trues = [0 for i in range(len(failure_labels))]
num_errors = 0
for x, y in zip(x_test, y_test):
    y_predict = estimator.predict([x])[0]
    if y != y_predict:
        errors[y][y_predict] += 1 
        num_errors += 1
        # print("%s predicted as %s" % (failure_labels[y], failure_labels[y_predict]), end=" = ")
        # print_sample(inverse_transform_sample(x, history), history)
    else: 
        trues[y] += 1 
        # print("%s predicted as %s" % (failure_labels[y], failure_labels[y_predict]), end=" = ")
        # print_sample(inverse_transform_sample(x, history), history)
    # if y == 4:
    #     print("%s predicted as %s" % (failure_labels[y], failure_labels[y_predict]), end=" = ")
    #     print_sample(inverse_transform_sample(x, history), history)
for y in range(len(failure_labels)):
    for y_predict in range(len(failure_labels)):
        if errors[y][y_predict] > 0:
            print("%s predicted as %s: %3d %2d%% | %2d%% %3d" % (failure_labels[y], failure_labels[y_predict], 
                errors[y][y_predict], (100.0*errors[y][y_predict])/num_errors, 
                (100.0*errors[y][y_predict])/(errors[y][y_predict]+trues[y]), errors[y][y_predict]+trues[y]))
