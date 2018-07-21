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
from sklearn.metrics import calinski_harabaz_score,silhouette_score 
def clear_title(title,remove_stopwords):
    raw_text=BeautifulSoup(title,'html').get_text()
    letters=re.sub('[^a-zA-Z]',' ',raw_text)
    words=letters.lower().split()
    if remove_stopwords:
	stop_words=set(stopwords.words('english'))
	words=[w for w in words if w not in stop_words]	
    return ' '.join(words)
dict_vec=DictVectorizer(sparse=False)
PATH_TO_ORIGINAL_DATA = '../../datasets/'
data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'fixed.csv', sep='\t')
selected_columns=['Product','Component','Assignee','Summary','Changed']
data=data[selected_columns]
classes=data['Assignee'].unique()
n_classes=len(classes)
classmap=pd.Series(data=np.arange(n_classes),index=classes)
data=pd.merge(data,pd.DataFrame({'Assignee':classes,'ClassId':classmap[classes].values}),on='Assignee',how='inner')
X_data=data[['Product','Component','Assignee','Summary','Changed']]
y_data=data['ClassId']
X_np_train,X_np_test,y_np_train,y_np_test=cross_validation.train_test_split(X_data,y_data,test_size=0.25)
X_train=pd.DataFrame(X_np_train)
X_test=pd.DataFrame(X_np_test)
y_train=pd.Series(y_np_train)
y_test=pd.Series(y_np_test)
X_train.columns=['Product','Component','Assignee','Summary','Changed']
X_dcr_train=X_train[['Product','Component']]
X_test.columns=['Product','Component','Assignee','Summary','Changed']
X_dcr_test=X_test[['Product','Component']]
X_dcr_train=dict_vec.fit_transform(X_dcr_train.to_dict(orient='records'))
X_dcr_test=dict_vec.transform(X_dcr_test.to_dict(orient='records'))
X_tfidf_train=[]
for title in X_train['Summary']:
    X_tfidf_train.append(clear_title(title,True))
X_tfidf_test=[]
for title in X_test['Summary']:
    X_tfidf_test.append(clear_title(title,True))
tfidf_vec=TfidfVectorizer(analyzer='word')
X_tfidf_train=tfidf_vec.fit_transform(X_tfidf_train)
X_tfidf_test=tfidf_vec.transform(X_tfidf_test)
def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print
    #打印主题-词语分布矩阵
    #print model.components_.shape
    #print model.components_
#doc_topic_prior=[0.001, 0.01, 0.05, 0.1, 0.2,0.5]
#topic_word_prior=[0.001, 0.01, 0.05, 0.1, 0.2,0.5]
#topics=[50,100,500,1000]
#iters=[50,100,500,1000]
#plex=[]
lda=LatentDirichletAllocation(n_components=100,max_iter=100,learning_method='batch',doc_topic_prior=0.5,topic_word_prior=0.2)
lda_begin_time=time.time()
lda.fit(X_tfidf_train)
lda_end_time=time.time()
print "LDA training time:%fs" % (lda_end_time-lda_begin_time)
X_tfidf_train=lda.transform(X_tfidf_train)
X_tfidf_test=lda.transform(X_tfidf_test)
X_train=np.concatenate((X_dcr_train,X_tfidf_train),axis=1)
#km=KMeans(n_clusters=30)
#km_begin_time=time.time()
#km.fit(X_train)
#km_end_time=time.time()
#print "KMeans training time:%fs" % (km_end_time-km_begin_time)
#print calinski_harabaz_score(X_train,km.labels_)
#print km.labels_
#ms=MeanShift()
#ms_begin_time=time.time()
#ms.fit(X_train)
#ms_end_time=time.time()
#print "MeanShift training time:%fs" % (ms_end_time-ms_begin_time)
#print calinski_harabaz_score(X_train,ms.labels_)
#print ms.labels_
#af=AffinityPropagation()
#af_begin_time=time.time()
#af.fit(X_train)
#af_end_time=time.time()
#print "AffinityPropagation training time:%fs" % (af_end_time-af_begin_time)
#print calinski_harabaz_score(X_train,af.labels_)
#print af.labels_
clusters=range(100,1100,100)
score_ac=[]
score_km=[]
for i in clusters:
    #ac=AgglomerativeClustering(n_clusters=i)
    #ac_begin_time=time.time()
    #ac.fit(X_train)
    #ac_end_time=time.time()
    #print "AgglomerativeClustering(%d) training time:%fs" % (i,ac_end_time-ac_begin_time)
    #score_ac.append(calinski_harabaz_score(X_train,ac.labels_))
    km=KMeans(n_clusters=i)
    km_begin_time=time.time()
    km.fit(X_train)
    km_end_time=time.time()
    print "KMeans(%d) training time:%fs" % (i,km_end_time-km_begin_time)
    score_km.append(calinski_harabaz_score(X_train,km.labels_))
X=clusters
#Y1=score_ac
Y2=score_km
print Y2
#plt.plot(X,Y1,color='blue',label='AgglomerativeClustering')
plt.plot(X,Y2,'o-',color='red',label='KMeans')
plt.xlabel('The value of k')
plt.ylabel('calinski_harabaz_score')
plt.legend()
plt.show()
