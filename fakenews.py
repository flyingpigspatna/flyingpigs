# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:25:31 2017

@author: KINGHSHUK
"""

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Merge, Dropout, add, concatenate, Reshape
from keras.optimizers import SGD
from keras.models import Sequential
import gensim as gs
import nltk
import numpy as np
from nltk.corpus import stopwords
import random
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)


maxLenofhead=30
maxLenofbody=2600

# Creating first CNN for TEXT
# CNN contains one convolution layer, maxpooling layer, dense, dropout, dense
cnn1=Sequential()
cnn1.add(Conv2D(64,(2,300), activation = 'relu',input_shape = (maxLenofhead,300,1)))
#cnn1.add(LeakyReLU(alpha=.01))
cnn1.add(MaxPooling2D(1,3))
#cnn1.add(Conv2D(128,(4,1), activation = 'relu'))
#cnn1.add(MaxPooling2D(1,3))
#cnn1.summary()
cnn1.add(Flatten())
cnn1.add(Dense(100, activation = 'relu'))
cnn1.add(Reshape((100,1)))
cnn1.add(LSTM(100, return_sequences = False))
cnn1.add(Dense(100, activation = 'sigmoid'))

# Creating second CNN for HYPOTHESIS
# CNN contains one convolution layer, maxpooling layer, dense, dropout, dense
cnn2=Sequential()
cnn2.add(Conv2D(64,(2,300), activation = 'relu',input_shape = (maxLenofbody,300,1)))
#cnn2.add(LeakyReLU(alpha=.01))
#cnn2.add(MaxPooling2D(1,5))
#cnn2.add(Conv2D(128,(4,1), activation = 'relu'))
cnn2.add(MaxPooling2D(1,3))
cnn2.add(Flatten())
cnn2.add(Dense(100, activation = 'relu'))
cnn2.add(Reshape((100,1)))
cnn2.add(LSTM(100, return_sequences = False))
cnn2.add(Dense(100, activation = 'sigmoid'))
# Joining two CNN and connecting with a ANN
classifier2=Sequential()
classifier2.add(Merge([cnn1,cnn2], mode='concat'))
# =============================================================================
# classifier2.add(Dense(70,activation='sigmoid'))
# classifier2.add(Dropout(0.2))
classifier2.add(Dense(65,activation='sigmoid'))
classifier2.add(Dropout(0.2))
classifier2.add(Dense(60,activation='sigmoid'))
classifier2.add(Dropout(0.2))
classifier2.add(Dense(1,activation='sigmoid'))

sgd = SGD(lr = 0.01, momentum = 0.9, decay=1e-2, nesterov = False)
classifier2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
r=random.random()
fl=open("./prediction.txt","w" )
for iterr in range(5):
    X=[]
    Y=[]
    
    # All the vectors of text
    X_train=[]
    
    # All the vectors of hypothesis
    Y_train=[]
    
    # All the vector of output
    # [1,0,0] for Entailment
    # [0,1,0] for Contradiction
    # [0,0,1] for Unknown
    tot_y_train=[]
    
    o=0
    
    with  open("./processed.txt" ) as file:
        for line in file:
            X=[]
            Y=[]
            q=line.strip().split("\t")
            tokens = nltk.word_tokenize(q[0].strip())
            for i in tokens:
                if i.lower() in model.wv.vocab:    
                    x=model.wv[i.lower()]
                else:
                    x=[0]*300
                X.append(x)
            X_train.append(X)
            tokens = nltk.word_tokenize(q[1].strip())
            for i in tokens:
                if i.lower() in model.wv.vocab:    
                    x=model.wv[i.lower()]
                else:
                    x=[0]*300
                Y.append(x)
            Y_train.append(Y)
            if(q[2].strip()=="unrelated"):
                tot_y_train.append([1])
            else:
                tot_y_train.append([0])
            o+=1
    
    # Start of training
    x_tot=[]
    y_tot=[]
    trn=[]
    random.shuffle(X_train, lambda : r)
    random.shuffle(Y_train, lambda : r)
    random.shuffle(tot_y_train, lambda : r)
    l=(len(X_train))
    for i in range(l):
        print("Total training:  "+str(i+1))
        y=len(X_train[i])
        x_t=X_train[i]
        for j in range(y,maxLenofhead):
            q=[0]*300
            x_t.append(q)
        y=len(Y_train[i])
        y_t=Y_train[i]
        for j in range(y,maxLenofbody):
            q=[0]*300
            y_t.append(q)
        x_tot.append(x_t)
        y_tot.append(y_t)
        y_train=tot_y_train[i]*1
        trn.append(y_train)
    
    x_ta=np.array(x_tot)
    
    # Reshaping the Text Vector
    x_ta=np.reshape(x_ta,(l,maxLenofhead,300,1))
    
    y_ta=np.array(y_tot)

    # Reshaping the Hypothesis vector
    y_ta=np.reshape(y_ta,(l,maxLenofbody,300,1))
    y_train=np.array(trn)
    y_train=np.reshape(y_train,(l,1))
    
    # Training the model
    classifier2.fit([x_ta, y_ta], y_train, batch_size = 1, epochs = 50, verbose = 1)
        
    
    # Start of testing
    # =============================================================================
    # l=len(X_train)
    # =============================================================================
        
    X=[]
    Y=[]
    X_train=[]
    # All the vectors of text
    X_train=[]
    Y_train=[]
    # All the vectors of hypothesis
    Y_train=[]
    
    # All the vector of output
    # [1,0,0] for Entailment
    # [0,1,0] for Contradiction
    # [0,0,1] for Unknown
    tot_y_train=[]
    tot_y_train=[]
    
    o=0
    
    with  open("./processed_test.txt" ) as file:
        for line in file:
            X=[]
            Y=[]
            q=line.strip().split("\t")
            tokens = nltk.word_tokenize(q[0].strip())
            for i in tokens:
                if i.lower() in model.wv.vocab:    
                    x=model.wv[i.lower()]
                else:
                    x=[0]*300
                X.append(x)
            X_train.append(X)
            tokens = nltk.word_tokenize(q[1].strip())
            for i in tokens:
                if i.lower() in model.wv.vocab:    
                    x=model.wv[i.lower()]
                else:
                    x=[0]*300
                Y.append(x)
            Y_train.append(Y)
            if(q[2].strip()=="unrelated"):
                tot_y_train.append([1])
            else:
                tot_y_train.append([0])
            o+=1
    tot=0

    for i in range(len(X_train)):
        y=len(X_train[i])
        x_t=X_train[i]
        for j in range(y,maxLenofhead):
            q=[0]*300
            x_t.append(q)
        y=len(Y_train[i])
        y_t=Y_train[i]
        for j in range(y,maxLenofbody):
            q=[0]*300
            y_t.append(q) 
        x_ta=np.array(x_t)
    
        x_ta=np.reshape(x_ta,(1,maxLenofhead,300,1))
        y_train=tot_y_train[i]*1
        y_ta=np.array(y_t)
    
        y_ta=np.reshape(y_ta,(1,maxLenofbody,300,1))
        y_train=np.array(y_train)
        y_train=np.reshape(y_train,(1,1))
        score = classifier2.evaluate([x_ta,y_ta], y_train, batch_size=1, verbose=1)
        val=classifier2.predict([x_ta,y_ta],batch_size=1,verbose=1)
    # =============================================================================
    #     fl.write(map(str,val))
    #     fl.write("\n")
    # =============================================================================
        print("\n%s: %.9f%%" % (classifier2.metrics_names[1], score[1]*100))
        tot=tot+int(score[1]*100)
    # Accuracy score
        
    sc=tot/len(X_train)
    fl.write(str(sc))
    fl.write("\n")
fl.close()
classifier2.save('cnn_lstm.h5')