#*- coding: utf-8 -*-
from sklearn.cluster import DBSCAN,KMeans,AffinityPropagation,MeanShift,AgglomerativeClustering
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
import string
from nltk.stem.porter import PorterStemmer
import theano
from theano import tensor as T
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.pipeline import Pipeline
import  matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
#preprocess the text
def clear_title(title,remove_stopwords):
    raw_text=BeautifulSoup(title,'html').get_text()   #filter the 'html' mark
    letters=re.sub('[^a-zA-Z]',' ',raw_text)        #filter the non-letter char
    words=letters.lower().split()            
    if remove_stopwords:                      #whether remove the stopwords or not
	stop_words=set(stopwords.words('english'))
	words=[w for w in words if w not in stop_words]	
    return ' '.join(words)
#read the file
PATH_TO_ORIGINAL_DATA = '../datasets/'
data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'fixed.csv',sep='\t')       #the dataset with bug reports that are all fixed
f_data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'data.csv',sep='\t')      #the dataset with bug reports whose status are REOPEN ,WONTFIX ,INVALID etc.
selected_columns=['Product','Component','Assignee','Summary','Changed']
data=data[selected_columns]
#number the classes of assignees for convinent training
classes=data['Assignee'].unique()
n_classes=len(classes)
classmap=pd.Series(data=np.arange(n_classes),index=classes)
data=pd.merge(data,pd.DataFrame({'Assignee':classes,'ClassId':classmap[classes].values}),on='Assignee',how='inner')
X_data=data[['Product','Component','Assignee','Summary','Changed']]
y_data=data['ClassId'] #the feature name of class numbered
#create a dict(key=assignee,value=the features of the assignee)
ass_dict={}
for ass in f_data['Assignee']:
    if not ass_dict.has_key(ass):
        ass_dict[ass]=[0,0,0]
for i in range(len(f_data)):
    ass=f_data['Assignee'][i]
    create=time.mktime(time.strptime(f_data['Create'][i],'%Y-%m-%d %H:%M:%S'))
    changed=time.mktime(time.strptime(f_data['Changed'][i],'%Y-%m-%d %H:%M:%S'))
    t=changed-create
    ass_dict[ass][0]+=1
    ass_dict[ass][1]+=t
    if f_data['Resolution'][i]=='FIXED':
        ass_dict[ass][2]+=1
total=float(len(f_data))
X_np_train,X_np_test,y_np_train,y_np_test=cross_validation.train_test_split(X_data,y_data,test_size=0.25) #split the dataset into training and testing with 3:1
X_train=pd.DataFrame(X_np_train)
X_test=pd.DataFrame(X_np_test)
y_train=pd.Series(y_np_train)
y_test=pd.Series(y_np_test)
#X_dcr_XXX means the discrete features in bug report dataset
X_train.columns=['Product','Component','Assignee','Summary','Changed']
X_dcr_train=X_train[['Product','Component']]
X_test.columns=['Product','Component','Assignee','Summary','Changed']
X_dcr_test=X_test[['Product','Component']]
#transform the discrete features into One-hot vectors
dict_vec=DictVectorizer(sparse=False)
X_dcr_train=dict_vec.fit_transform(X_dcr_train.to_dict(orient='records'))
X_dcr_test=dict_vec.transform(X_dcr_test.to_dict(orient='records'))
X_tfidf_train=[]
for title in X_train['Summary']:
    X_tfidf_train.append(clear_title(title,True))
X_tfidf_test=[]
for title in X_test['Summary']:
    X_tfidf_test.append(clear_title(title,True))
#transform the text into tfidf vectors
tfidf_vec=TfidfVectorizer(analyzer='word')
X_tfidf_train=tfidf_vec.fit_transform(X_tfidf_train)
X_tfidf_test=tfidf_vec.transform(X_tfidf_test)
def print_top_words(model, feature_names, n_top_words):
    #print terms with heavier weigtht in each topic
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print
    #print topics-term distributed matrix
    #print model.components_.shape
    #print model.components_
#train the lda model
lda=LatentDirichletAllocation(n_components=180,max_iter=100,learning_method='batch',doc_topic_prior=0.5,topic_word_prior=0.2)
lda_begin_time=time.time()
lda.fit(X_tfidf_train)
lda_end_time=time.time()
print "LDA training time:%fs" % (lda_end_time-lda_begin_time)
#transform the text into topic vectors
X_tfidf_train=lda.transform(X_tfidf_train)
X_tfidf_test=lda.transform(X_tfidf_test)
#concatenate the feature vectors
X_train=np.concatenate((X_dcr_train,X_tfidf_train),axis=1)
X_test=np.concatenate((X_dcr_test,X_tfidf_test),axis=1)
#train the k-means++ model
km=KMeans(n_clusters=500)
km_begin_time=time.time()
km.fit(X_train)
km_end_time=time.time()
print "KMeans training time:%fs" % (km_end_time-km_begin_time)
train_labels=km.labels_.reshape(len(km.labels_),1)
tmp_labels=km.predict(X_test)
test_labels=tmp_labels.reshape(len(tmp_labels),1)
X_labels_train=np.concatenate((X_np_train,train_labels),axis=1)
X_labels_test=np.concatenate((X_np_test,test_labels),axis=1)
#save the results with the time order after clustered 
train_assignee_dict={}
train_report_dict={}
for report in X_labels_train:
    train_report_dict[report[5]]=[]
    train_assignee_dict[report[2]]=1
for report in X_labels_train:
    if train_report_dict.has_key(report[5]):
        t=time.mktime(time.strptime(report[4],'%Y-%m-%d %H:%M:%S'))
        train_report_dict[report[5]].append(tuple((str(t),report[2])))
for key in train_report_dict:
    train_report_dict[key].sort()
with open('../rec_module/report_labels_train','w') as f:
    f.write('SessionId\tItemId\tActive\tSkill\tTime\n')
    for key in train_report_dict:
        for item in train_report_dict[key]:
            active=ass_dict[item[1]][0]
            tmp=1/(ass_dict[item[1]][1]/total)
            repair_time=(tmp-1.8033736213e-07)/(5.364150943399999-1.8033736213e-07)         #normalize the work efficiency
            vict=(active-1)/629.0                                                           #normalize the bug reports repaired successfully
            skill=(repair_time*0.8+vict*0.2)/2  
            f.write(str(key)+'\t'+item[1]+'\t'+str(active/total)+'\t'+str(skill)+'\t'+item[0])
            f.write('\n')
#process the testing set like training set
test_report_dict={}
for report in X_labels_test:
    test_report_dict[report[5]]=[]
for report in X_labels_test:
    if test_report_dict.has_key(report[5]):
        t=time.mktime(time.strptime(report[4],'%Y-%m-%d %H:%M:%S'))
        test_report_dict[report[5]].append(tuple((str(t),report[2])))
for key in test_report_dict:
    test_report_dict[key].sort()
with open('../rec_module/report_labels_test','w') as f:
    f.write('SessionId\tItemId\tActive\tSkill\tTime\n')
    for key in test_report_dict:
        for item in test_report_dict[key]:
            if train_assignee_dict.has_key(item[1]):
                active=ass_dict[item[1]][0]
                tmp=1/(ass_dict[item[1]][1]/total)
                repair_time=(tmp-1.8033736213e-07)/(5.364150943399999-1.8033736213e-07)        
                vict=(active-1)/629.0
                skill=(repair_time*0.8+vict*0.2)/2  
                f.write(str(key)+'\t'+item[1]+'\t'+str(active/total)+'\t'+str(skill)+'\t'+item[0])
                f.write('\n')
