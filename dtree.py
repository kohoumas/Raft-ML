#!/usr/bin/python3

import sys
import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
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

failure_labels_old, dataset_old = enhance_failure_labels_play_old(dataset.copy())
failure_labels, dataset = enhance_failure_labels_play(dataset)
depicted_columns = add_suffixes(history, 
    ['<0-to2-#','<0-to3-#','<0-to4-#','<0-from2-#','<0-from3-#','<0-from4-#',
     '8-to2-#','8-to3-#','8-to4-#','8-from2-#','8-from3-#','8-from4-#',
     '9-to2-#','9-to3-#','9-to4-#','9-from2-#','9-from3-#','9-from4-#',
     '5-to2-#','5-to3-#','5-to4-#','5-from2-#','5-from3-#','5-from4-#',
     '6-to2-#','6-to3-#','6-to4-#','6-from2-#','6-from3-#','6-from4-#',
     '7-to2-#','7-to3-#','7-to4-#','7-from2-#','7-from3-#','7-from4-#'])
depicted_columns += add_suffixes(history, ['time'])
#depicted_columns += ["prevLeader1", "prevLeader2"]
depicted_columns += ["SumTime", "VarTime"]
depicted_columns += ['Num<0-to2','Num<0-to3','Num<0-to4','Num<0-from2','Num<0-from3','Num<0-from4',
     'Num8-to2','Num8-to3','Num8-to4','Num8-from2','Num8-from3','Num8-from4',
     'Num9-to2','Num9-to3','Num9-to4','Num9-from2','Num9-from3','Num9-from4',
     'Num5-to2','Num5-to3','Num5-to4','Num5-from2','Num5-from3','Num5-from4',
     'Num6-to2','Num6-to3','Num6-to4','Num6-from2','Num6-from3','Num6-from4',
     'Num7-to2','Num7-to3','Num7-to4','Num7-from2','Num7-from3','Num7-from4']
depicted_columns += ['Last<0-to2','Last<0-to3','Last<0-to4','Last<0-from2','Last<0-from3','Last<0-from4',
     'Last8-to2','Last8-to3','Last8-to4','Last8-from2','Last8-from3','Last8-from4',
     'Last9-to2','Last9-to3','Last9-to4','Last9-from2','Last9-from3','Last9-from4',
     'Last5-to2','Last5-to3','Last5-to4','Last5-from2','Last5-from3','Last5-from4',
     'Last6-to2','Last6-to3','Last6-to4','Last6-from2','Last6-from3','Last6-from4',
     'Last7-to2','Last7-to3','Last7-to4','Last7-from2','Last7-from3','Last7-from4']
depicted_columns += ['First<0-to2','First<0-to3','First<0-to4','First<0-from2','First<0-from3','First<0-from4',
     'First8-to2','First8-to3','First8-to4','First8-from2','First8-from3','First8-from4',
     'First9-to2','First9-to3','First9-to4','First9-from2','First9-from3','First9-from4',
     'First5-to2','First5-to3','First5-to4','First5-from2','First5-from3','First5-from4',
     'First6-to2','First6-to3','First6-to4','First6-from2','First6-from3','First6-from4',
     'First7-to2','First7-to3','First7-to4','First7-from2','First7-from3','First7-from4']
depicted_columns += ['Num-fromto2','Num-fromto3','Num-fromto4']
depicted_columns += ['Last-fromto2','Last-fromto3','Last-fromto4']
depicted_columns += ['First-fromto2','First-fromto3','First-fromto4']
depicted_columns += ['Num8']

# filter same labels
filter_label_indexes = range(2) #range(6) # all indexes for the 'play' version
#filter_label_indexes = [index for index in range(len(failure_labels)) if index not in filter_label_indexes] # the complement set of indexes
filter_labels = [failure_labels[index] for index in filter_label_indexes]
dataset = dataset[dataset.failure.isin(filter_label_indexes)]
for index, label in enumerate(filter_labels):
    dataset.loc[(dataset.failure == failure_labels.index(label)), 'failure'] = index
failure_labels = filter_labels

data = dataset[columns]
data = data.values
transformed_data = [transform_sample(sample, history, binary=True) for sample in data]
target = dataset['failure']
target = target.values
target_old = dataset_old['failure']
print(len(target_old))
print(len(target_old[target_old==0])/len(target_old)*100)
print(len(target_old[target_old==1])/len(target_old)*100)
print(len(target_old[target_old==2])/len(target_old)*100)
print(len(target_old[target_old==3])/len(target_old)*100)
print(len(target_old[target_old==4])/len(target_old)*100)
print(len(target_old[target_old==5])/len(target_old)*100)
target_old = target_old.values

impurity_factor = 0.01
depth = None

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(transformed_data, target, target_old, test_size=0.1, shuffle=True)

estimator = DecisionTreeClassifier(splitter='best', max_depth=depth, max_features=None, min_impurity_decrease=impurity_factor, class_weight='balanced')
estimator.fit(x_train, y_train)
#dump(estimator, 'dtree.joblib') # saves estimator to a file

# Testing
y_pred = estimator.predict(x_train)
score = accuracy_score(y_train,y_pred)
score = int(100*score)
print("score on training = %d%%" % score)
y_pred = estimator.predict(x_test)
score = accuracy_score(y_test,y_pred)
score = int(100*score)
print("score on testing = %d%%" % score)

# Draw graph
trained_labels = []
for label_index in range(len(failure_labels)):
    if label_index in y_train:
        print("failure %s is included in training" % failure_labels[label_index])
        trained_labels.append(failure_labels[label_index])
dot_data = export_graphviz(decision_tree=estimator, out_file=None, rounded=True, 
                                feature_names=depicted_columns,  
                                class_names=trained_labels,
                                filled=True)
graph = graphviz.Source(dot_data, format="png")
#time = time.time()
graph.render("NEW_PNGS/" + dataset_filename[:-4] + "_impurity" + str(impurity_factor) + "_score" + str(score))# + "_depth" + str(depth)) # + str(time))

errors = [[0] * len(failure_labels) for i in range(len(failure_labels))]
trues = [0] * len(failure_labels)
num_errors = 0
for x, y, z in zip(x_test, y_test, z_test):
    y_predict = estimator.predict([x])[0]
    if y != y_predict:
        errors[y][y_predict] += 1 
        num_errors += 1
        #print("%s predicted as %s" % (failure_labels[y], failure_labels[y_predict]), end=" = ")
        print("%s (or %s) predicted as %s" % (failure_labels_old[z], failure_labels[y], failure_labels[y_predict]), end=" = ")
       	print_sample(inverse_transform_sample(x, history), history)
    else: 
        trues[y] += 1 
        #print("%s predicted as %s" % (failure_labels[y], failure_labels[y_predict]), end=" = ")
        #print_sample(inverse_transform_sample(x, history), history)
for y in range(len(failure_labels)):
    for y_predict in range(len(failure_labels)):
        if errors[y][y_predict] > 0:
            print("%s predicted as %s: %3d %2d%% | %2d%% %3d" % (failure_labels[y], failure_labels[y_predict], 
                errors[y][y_predict], (100.0*errors[y][y_predict])/num_errors, 
                (100.0*errors[y][y_predict])/(errors[y][y_predict]+trues[y]), errors[y][y_predict]+trues[y]))

leaf_classes_old = {}
leafs = estimator.apply(x_train)
for z, leaf in zip(z_train, leafs):
    if leaf in leaf_classes_old:
        leaf_classes_old[leaf][z] += 1
    else: 
        leaf_classes_old[leaf] = [0] * len(failure_labels_old)
for leaf in leaf_classes_old:
    print(leaf, end = " : ")
    print(leaf_classes_old[leaf])
#print(estimator.tree_.n_node_samples[0])
