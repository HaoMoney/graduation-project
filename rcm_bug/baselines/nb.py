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
#X_lda_train=[]
#for title in X_train['Summary']:
#    X_lda_train.append(textPrecessing(title))
#with open('text','w') as f:
#    for line in X_tfidf_train:
#  	f.write(line+'\n')
#dic=Dictionary(X_lda_train)		
#bag_of_words=[dic.doc2bow(w) for w in X_lda_train]
#lda=LdaModel(bag_of_words,num_topics=10,id2word=dic,passes=5)
#lda.print_topics(10)
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
#lda=LatentDirichletAllocation(n_components=180,max_iter=100,learning_method='batch',doc_topic_prior=0.01,topic_word_prior=0.01)
#X_tmp=lda.fit_transform(X_tfidf_train)
#tfidf_feature_names=tfidf_vec.get_feature_names()
#X_tfidf_train=list(X_tfidf_train)
X_tfidf_test=tfidf_vec.transform(X_tfidf_test)
#pca=PCA(n_components=600)
#X_tfidf_train=pca.fit_transform(X_tfidf_train.toarray())
#X_tfidf_test=pca.transform(X_tfidf_test.toarray())
#print X_tfidf_train
#print X_tfidf_train.shape
#print type(X_dcr_train)
#class Layer(object):
#    def __init__(self, inputs, in_size, out_size, activation_function=None):
#        self.w = theano.shared(np.random.randn((in_size, out_size)))
#        self.b = theano.shared(np.zeros(out_size))
#        self.Wx_plus_b = T.dot(inputs, self.w) + self.b
#        self.activation_function = activation_function
#        if activation_function is None:
#            self.outputs = self.Wx_plus_b
#        else:
#            self.outputs = self.activation_function(self.Wx_plus_b)
#X_train=np.concatenate((X_dcr_train,X_tfidf_train.toarray()),axis=1)
#X_test=np.concatenate((X_dcr_test,X_tfidf_test.toarray()),axis=1)
#feats=len(X_train[0])
#nn_input_dim=feats #输入神经元个数  
#nn_output_dim=n_classes #输出神经元个数  
#nn_hdim=200 
#x=T.dmatrix("x")
#y=T.ivector("y")
#w1=theano.shared(np.random.randn(nn_input_dim,nn_hdim),name="w1")  
#b1=theano.shared(np.zeros(nn_hdim),name="b1")
#print w1  
#w2=theano.shared(np.random.randn(nn_hdim,nn_output_dim),name="w2")  
#b2=theano.shared(np.zeros(nn_output_dim),name="b2")
#z1=x.dot(w1)+b1   #1  
#a1=T.tanh(z1)     #2  
#z2=a1.dot(w2)+b2  #3  
#y_hat=T.nnet.softmax(z2)
#loss=T.nnet.categorical_crossentropy(y_hat,y).mean()
#prediction=T.argmax(y_hat,axis=1)
#forword_prop=theano.function([x],y_hat)  
#calculate_loss=theano.function([x,y],loss)  
#predict=theano.function([x],prediction)
#epsilon=0.01
##求导  
#dw2=T.grad(loss,w1)  
#db2=T.grad(loss,b1)  
#dw1=T.grad(loss,w2)  
#db1=T.grad(loss,b2)  
#  
##更新值  
#gradient_step=theano.function(  
#    [x,y],  
#    updates=(  
#        (w2,w2-epsilon*dw2),  
#        (b2,b2-epsilon*db2),  
#        (w1,w1-epsilon*dw1),  
#        (b1,b1-epsilon*db1)  
#  
#    )  
#)
#for i in xrange(0,1000):  
#    gradient_step(X_train,y_train)  
#    if print_loss and i%1000==0:  
#        print "Loss after iteration %i: %f" %(i,calculate_loss(X_train,y_train))
#dnn_classifier=MLPClassifier(hidden_layer_sizes=(4000,500,n_classes),learning_rate_init=0.01,early_stopping=True)
#dnn_begin_time=time.time()
#dnn_classifier.fit(X_train,y_train)
#dnn_end_time=time.time()
#print "DNN training time:%fs" % (dnn_end_time-dnn_begin_time)
#dnn_y_pred=dnn_classifier.predict(X_test)
#dnn_acc=accuracy_score(y_test,dnn_y_pred)
#print "DNN accuracy:%f" % (dnn_acc)
#pip_tfidf.fit(X_text_train,y_train)
#y_text_pred=svc.predict(X_text_test)
#text_acc=accuracy_score(y_test,y_text_pred)
#X_test=pd.DataFrame(X_test)
#X_test.columns=['Product','Component']
#y_train=pd.Series(y_train)
#print y_train
#y_train.columns=['Assignee']
#print y_train.to_dict()
#y_test=pd.Series(y_test)
#y_train=dict_vec.transform(y_train.to_dict())
#print y_train
#X_test=dict_vec.transform(X_test.to_dict(outtype='records'))
#y_test=dict_vec.transform(y_test.to_dict())
#X_in_test=pca.transform(X_in_test)
#print X_tfidf_train.shape
#print X_in_test.shape
#svc=SVC(kernel='linear')
#start1=time.time()
#svc.fit(X_train,y_train)
#end1=time.time()
#inter_svc=end1-start1
mnb=MultinomialNB()
#start2=time.time()
mnb.fit(X_tfidf_train,y_train)
#end2=time.time()
#inter_mnb=end2-start2
#print "Training time:"
#print "svc:%fs,mnb:%fs" % (inter_svc,inter_mnb)
#svc_y_pred=svc.predict(X_test)
#svc_acc=accuracy_score(y_test,svc_y_pred)
mnb_y_pred=mnb.predict(X_tfidf_test)
mnb_acc=accuracy_score(y_test,mnb_y_pred)
print "mnb:%f" % (mnb_acc)
