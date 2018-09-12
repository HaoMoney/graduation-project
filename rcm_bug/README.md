###1. Preprocess the dataset

#Two files: my_xml.py & preprocess.py

'my_xml.py' is used to preprocess the xml files that comes from the bug tracking system

'preprocess.py' is used to filter the invalid fields

2. Run the baselines 

#The directory: baseline

If you want to run one of them , just excute : 'python XXX.py'

For example,if you want to run the SVM method , please excute : 'python svm.py'

#Attention:

The deep learning method is in directory : 'deep_learning'

#Two types of word vectors: Glove.6B.50d.txt & summary.txt

'Glove.6B.50d.txt' is from Stanford , 'summary.txt' is from training the Word2Vec model with the bug report dataset. 

If you want to run this baseline , please excute 'python train_XXX.py'

For example,if you want to run the CNN method , please excute 'python train_CNN.py'

3. Two modules : cluster modulle & recommendation module

Our codes include two modules : cluster module and recommendation module

First,you need to cluser the dataset ,please open the cluster_module and excute the 'cluster.py'

After clustered , it will generate two files: 'report_labels_train' & 'report_labels_test'

'report_labels_train' is for training data , 'report_labels_test' is for testing data.

Both of them will be in the directory : 'rec_module'

Second , you need to open the directory : 'rec_module'

This module is based on the algorithm of "Session-based Recommendations With Recurrent Neural Networks". See paper: http://arxiv.org/abs/1511.06939

If you want to run this module without the features of assignees , please excute 'python run.py'

If you want to run this module with the features of assignees , please excute 'python run_added.py'

The results include two measures: Top-k accuracy(accuracy@k) & MRR

#P.S.

Our codes are excuting in three dataset : Eclipse , Mozilla , Open Office

But in the codes ,we only provide the Eclipse dataset.

If you want to test other dataset , please download by yourself.

'data.csv' is bug reports whose status are REOPEN ,WONTFIX ,INVALID etc.

'fixed.csv' is bug reports that are all fixed

In the cluster_module , the directory 'params' is used to adjust the parameters so you can just ignore.
