# Raft-ML
Machine Learning for enhancing etcd/Raft

More details in paper of Kostas Choumas and Thanasis Korakis,
"When Machine Learning Meets Raft: How to Elect a Leader over a Network", Proceedings of IEEE GLOBECOM 2023.

How to use:

1. Run 'python mininet-raft-3nodes.py > meta-data.txt' (or 'mininet-raft-5nodes.py') to deploy a Raft cluster with 3 (or 5) nodes, after installing the Mininet environment. 
Additionally, pyshark must be installed in order to generate pcap files containing the captured packets, those engaged in the communication between the Raft nodes.
The output of this execution is stored in a file named 'meta-data.txt'.

2. Run 'python3 pyshark_read_raft.py > data-output-1.txt' to convert the 'raft-output-1.pcap' to 'data-output-1.txt'.

3. Run 'python3 dataset_preprocessing.py 1000 data-output-1.txt' to create a file named 'dataset_output_...csv', with 1000 samples. It uses the 'meta-data.txt' file. If the leadership transitions are less than 1000, you should replace 1000 with the lower number. 

4. Run 'python3 dtree.py dataset_output_...csv' to train a Decision Tree model on this dataset. Similary, you can train a SVM or KNN model, all are based on supervised classifiers of scikit-learn.
