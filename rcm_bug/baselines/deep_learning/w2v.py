#*- coding: utf-8 -*-
"""
@author: Hao Qian
"""
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
data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'cleared_bugs',sep='\t')
selected_columns=['Product','Component','Assignee','Summary','Changed']
data=data[selected_columns]
text=[]
for title in data['Summary']:
    text.append(clear_title(title,True).split())
print text
from gensim.models import word2vec
model=word2vec.Word2Vec(text,workers=4,size=50,min_count=1,window=2)
model.wv.save_word2vec_format('summary.txt',binary=False)
