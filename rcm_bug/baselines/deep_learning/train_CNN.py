#*- coding: utf-8 -*-
from __future__ import print_function
"""
@author: Hao Qian
"""
'''This script loads pre-trained word embeddings (GloVe embeddings)
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
'''
import os
import numpy as np
np.random.seed(1337)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding,Add,Concatenate
from keras.models import Model
from keras.optimizers import *
import sys
import pandas as pd
import keras.backend as K
#BASE_DIR = '.'
#GLOVE_DIR = BASE_DIR + '/glove.6B/'
MAX_SEQUENCE_LENGTH =50
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.25
BATCH=32
# first, build index mapping words in the embeddings set
# to their embedding vector
#print('Indexing word vectors.')
embeddings_index = {}
#f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
f = open(os.path.join('summary.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
# second, prepare text samples and their labels
print('Processing text dataset')
# finally, vectorize the text samples into a 2D integer tensor
PATH_TO_ORIGINAL_DATA = '../../datasets/'
data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'fixed.csv', sep='\t')
selected_columns=['Assignee','Summary']
data=data[selected_columns]
classes=data['Assignee'].unique()
n_classes=len(classes)
classmap=pd.Series(data=np.arange(n_classes),index=classes)
data=pd.merge(data,pd.DataFrame({'Assignee':classes,'ClassId':classmap[classes].values}),on='Assignee',how='inner')
texts=data['Summary']
labels=data['ClassId']
print('Found %s texts.' % len(texts))
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
print('Preparing embedding matrix.')
# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<20000(nb_words)
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            trainable=True)
print('Training model.')
# train a 1D convnet with global maxpoolinnb_wordsg
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='tanh')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='tanh')(x)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
y = Conv1D(128, 4, activation='tanh')(embedded_sequences)
y = MaxPooling1D(4)(y)
y = Conv1D(128, 4, activation='tanh')(y)
y = MaxPooling1D(4)(y)
y = Flatten()(y)
z = Conv1D(128, 6, activation='tanh')(embedded_sequences)
z = MaxPooling1D(6)(z)
z = Flatten()(z)
added=Concatenate()([x,y,z])
m = Dense(128, activation='tanh')(added)
preds = Dense(n_classes, activation='softmax')(m)
def mrr(y_true,y_pred,k=5):
    #tmp=np.argsort(-y_pred)
    #b=tmp[:,:k]
    #c=np.argmax(y_true,axis=-1) 
    #for i in range(len(c)):
    #    for j in range(len(b[i])):
    #        if c[i]==b[i][j]:
    #            m.append(1.0/(j+1))
    #return np.mean(m)
    #m=K.argmax(y_true,axis=-1)
    true=tf.reshape(tf.cast(K.argmax(y_true,axis=-1),tf.int32),(BATCH,1))
    top_value,top_idx=tf.nn.top_k(y_pred,k)    
    ranks=tf.cast(tf.where(tf.equal(top_idx,true))[:,1],tf.float32)
    value=tf.reduce_mean(1.0/ranks+1)
    return value

def top_k_acc(y_true, y_pred, k=1):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=[top_k_acc])

model.fit(x_train, y_train,batch_size=BATCH,epochs=30,verbose=2)
score = model.evaluate(x_train, y_train,batch_size=BATCH,verbose=0) 
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate(x_val, y_val,batch_size=BATCH,verbose=0) 
print('Test score:', score[0])
print('Test accuracy:', score[1])
