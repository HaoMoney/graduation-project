#*- coding: utf-8 -*-
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
#from gensim.models import LdaModel
#from gensim.corpora import Dictionary
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
import string
from nltk.stem.porter import PorterStemmer
import theano
from theano import tensor as T
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
def textPrecessing(text):
    #小写化
    text = text.lower()
    #去除特殊标点
    for c in string.punctuation:
        text = text.replace(c, ' ')
    text=re.sub('[^a-z]',' ',text)
    #分词
    wordLst = nltk.word_tokenize(text)
    #去除停用词
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    #仅保留名词或特定POS   
    #refiltered =nltk.pos_tag(filtered)
    #filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    #词干化
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]
    return " ".join(filtered)
def clear_title(title,remove_stopwords):
    raw_text=BeautifulSoup(title,'html').get_text()
    letters=re.sub('[^a-zA-Z]',' ',raw_text)
    words=letters.lower().split()
    if remove_stopwords:
	stop_words=set(stopwords.words('english'))
	words=[w for w in words if w not in stop_words]	
    return ' '.join(words)
dict_vec=DictVectorizer(sparse=False)
PATH_TO_ORIGINAL_DATA = '../datasets/'
#PATH_TO_PROCESSED_DATA = '/path/to/store/processed/data/'
data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'cleared_openoffice', sep='\t')
selected_columns=['Product','Component','Assignee','Summary']
data=data[selected_columns]
#print len(data['Product'].unique())
#print len(data['Component'].unique())
classes=data['Assignee'].unique()
n_classes=len(classes)
#print n_classes
#print n_classes
classmap=pd.Series(data=np.arange(n_classes),index=classes)
#print classmap
data=pd.merge(data,pd.DataFrame({'Assignee':classes,'ClassId':classmap[classes].values}),on='Assignee',how='inner')
X_data=data[['Product','Component','Summary']]
y_data=data['ClassId']
#print y_data
#print data.info()
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X_data,y_data,test_size=0.25)
#print len(y_train)
#print len(y_test)
#print y_train
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
y_train=pd.Series(y_train)
y_test=pd.Series(y_test)
X_train.columns=['Product','Component','Summary']
X_dcr_train=X_train[['Product','Component']]
X_test.columns=['Product','Component','Summary']
X_dcr_test=X_test[['Product','Component']]
X_dcr_train=dict_vec.fit_transform(X_dcr_train.to_dict(orient='records'))
X_dcr_test=dict_vec.transform(X_dcr_test.to_dict(orient='records'))
X_tfidf_train=[]
for title in X_train['Summary']:
    X_tfidf_train.append(clear_title(title,True))
#print X_tfidf_train
X_tfidf_test=[]
for title in X_test['Summary']:
    X_tfidf_test.append(clear_title(title,True))
tfidf_vec=TfidfVectorizer(analyzer='word')
X_tfidf_train=tfidf_vec.fit_transform(X_tfidf_train)
#print X_tfidf_train
def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print
    #打印主题-词语分布矩阵
    #print model.components_.shape
    #print model.components_
#tfidf_feature_names=tfidf_vec.get_feature_names()
#X_tfidf_train=list(X_tfidf_train)
X_tfidf_test=tfidf_vec.transform(X_tfidf_test)
#X_train=np.concatenate((X_dcr_train,X_tfidf_train.toarray()),axis=1)
#X_test=np.concatenate((X_dcr_test,X_tfidf_test.toarray()),axis=1)
#lda=LatentDirichletAllocation(n_components=180,max_iter=100,learning_method='batch')
#lda_begin_time=time.time()
#lda.fit(X_tfidf_train)
#lda_end_time=time.time()
#print "LDA training time:%fs" % (lda_end_time-lda_begin_time)
#X_tfidf_train=lda.transform(X_tfidf_train)
#X_tfidf_test=lda.transform(X_tfidf_test)
#X_train=np.concatenate((X_dcr_train,X_tfidf_train),axis=1)
#X_test=np.concatenate((X_dcr_test,X_tfidf_test),axis=1)
lsvc=SVC(kernel='linear')
ssvc=SVC(kernel='sigmoid')
psvc=SVC(kernel='poly')
rsvc=SVC(kernel='rbf')
#start1=time.time()
lsvc.fit(X_tfidf_train,y_train)
ssvc.fit(X_tfidf_train,y_train)
psvc.fit(X_tfidf_train,y_train)
rsvc.fit(X_tfidf_train,y_train)
#end1=time.time()
#inter_svc=end1-start1
#mnb=MultinomialNB()
#start2=time.time()
#mnb.fit(X_train,y_train)
#end2=time.time()
#inter_mnb=end2-start2
#print "Training time:"
#print "svc:%fs,mnb:%fs" % (inter_svc,inter_mnb)
lsvc_y_pred=lsvc.predict(X_tfidf_test)
#ssvc_y_pred=ssvc.predict(X_tfidf_test)
#psvc_y_pred=psvc.predict(X_tfidf_test)
#rsvc_y_pred=rsvc.predict(X_tfidf_test)
lsvc_acc=accuracy_score(y_test,lsvc_y_pred)
#ssvc_acc=accuracy_score(y_test,ssvc_y_pred)
#psvc_acc=accuracy_score(y_test,psvc_y_pred)
#rsvc_acc=accuracy_score(y_test,rsvc_y_pred)
print "linear:%f" % (lsvc_acc)
#mnb_y_pred=mnb.predict(X_test)
#mnb_acc=accuracy_score(y_test,mnb_y_pred)
#print "Accuracy:svc:%f\tmnb:%f" % (svc_acc,mnb_acc)
