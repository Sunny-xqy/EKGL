import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from networkx.classes import graph
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
import time
import networkx as nx
from utils import *
import tensorflow as tf
from keras.layers import Input, Dense,  Concatenate,Dropout,Lambda,Multiply,Activation,Add,dot,RepeatVector
from keras.models import Model,Sequential
from keras.optimizers import Nadam

import math
# from node2vec import Node2Vec
from utils import *
from walk import RWGraph
import pickle



def myloss(y_true,y_pre):
    error = tf.abs(y_true - y_pre)
    # error = K.square(y_pre - y_true)
    if tf.maximum(error,0.01) == 0.01:
        return 0
    else:
        one = tf.ones_like(y_true)           
        zero = tf.zeros_like(y_true)
        y_true = tf.where(y_true <= 0.5, x=zero, y=one)
        y_pre = tf.where(y_pre <= 0.5 , x=zero, y=one)
        p = tf.exp(tf.abs(y_true - y_pre))
        return tf.sum(error*p)
    
def att(att,X):
    k = tf.softmax(att)
    attention = Lambda(lambda x:x*k)
    k = attention(X)
    k = tf.sum(k,axis=1)
    return k


def CoNN_model():
    model = Sequential()
    x1_in = Input(shape=(200,))
    x2_in = Input(shape=(200,))
    # x3 = Input(shape=(1,))
    
    X1 = Dense(200,activation='tanh')(x1_in)
    X2 = Dense(200,activation='tanh' )(x1_in)
    X3 = Dense(200,activation='tanh' )(x1_in)
    
   
    attention_1 =  dot([X1, x2_in],axes=1)
    attention_2 =  dot([X2, x2_in],axes=1)
    attention_3 =  dot([X3, x2_in],axes=1)
    
    # sum_ = Lambda(lambda x: K.sum(K.exp(x),axis=-1))
    attention = Concatenate(axis=1)([attention_1, attention_2, attention_3])
    attention = Activation('softmax',name='keys')(attention)
    attention = RepeatVector(200)(attention)
    
    w1 =  Lambda(lambda att:att[:,:,0] , name='keys1')
    w2 =  Lambda(lambda att:att[:,:,1], name = 'keys2')
    w3 = Lambda(lambda att:att[:,:,2], name = 'keys3')

    attention1 = w1(attention)
    attention2 = w2(attention)
    attention3 = w3(attention)

    X1 = Multiply(name='value1')([attention_1,attention1])
    X2 = Multiply(name = 'value2')([attention_2,attention2])
    X3 = Multiply(name = 'value3')([attention_3,attention3])

    t = Add()([X1,X2,X3])
    t = Dropout(0.5)(t)
    t = Dense(100,activation='relu',name='den_att1')(t)
    # t = LeakyReLU(alpha=0.3)(t)
    t = Dropout(0.5,name='dr_att1')(t)
    t = Dense(50, activation='relu',name='den_att2')(t)
    
    t = Dropout(0.5,name='dr_att2')(t)
    t = Dense(10,activation='relu' ,name='den_att3')(t)

    zz = Dense(1,activation = 'sigmoid',name='den_att4')(t)
    


    mul1 = dot([x1_in,x2_in],axes=1)
    mul1 = Activation('sigmoid')(mul1)
    XX = Concatenate(axis=1)([x1_in,mul1,x2_in])
    XX= Dropout(0.5)(XX)
    XX = Dense(10,activation='relu')(XX)
    XX =Dense(1, activation='sigmoid')(XX) 
    
    Y = Concatenate(axis=1)([zz,XX])
    Y = Dense(1,activation = 'linear')(Y)

    model = Model(inputs=[x1_in,x2_in],outputs=Y)

    model.compile(
        loss=myloss,
        # loss='mse',
        optimizer=Nadam(0.0003),
        # optimizer="sgd",#(0.01),
        metrics=['mean_squared_error']
    )
    model.summary()
    return model

def train_model(data,valid_data,batch_size,filepath):
    model = CoNN_model()
    # filepath='_{epoch:02d}-{loss:.3f}.h5'
    early_stopping = EarlyStopping(monitor='loss', patience=10,verbose=2) 
    plateau = ReduceLROnPlateau(monitor="val_loss", verbose=2, mode='min', factor=0.5, patience=5) 
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, period=1,
                                     save_best_only=True, mode='min', save_weights_only=False)  
    history = model.fit(
        data[0],data[1], 
        epochs=200,
        batch_size=batch_size,
        # steps_per_epoch=len(data[0]),
        callbacks=[early_stopping,plateau],
        validation_data =(valid_data[0],valid_data[1]))
        #verbose=2,
        #batch_size=64,
        # callbacks=[early_stopping, plateau,checkpoint,]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    metrics = history.history['mean_squared_error']
    val_metrics = history.history['val_mean_squared_error']
    model.save('model/banzhaf_EKGL.h5')
    return loss,val_loss,metrics,val_metrics
    # return loss,metrics

def read_enc():
    pkl_file = open('data/system_all.pkl','rb')
    enc =pickle.load(pkl_file)
    return enc

def getdata(embedding_result):
    data1 = []
    data2 = []
    data3 = []
    controlpower = []
    f1=open('data/groundtruthforcontroller.txt','r')
    datas = f1.readlines()
    for data in datas:
        links=data[:-1].split(';')
        enc = read_enc()
        for link in links[:-1]:
            try:
                node=link.split(',')
                node0 = embedding_result[enc.transform([node[0]])[0]]
                node1 = embedding_result[enc.transform([node[1]])[0]]
                node01= float(node[2])
                if node01 <= 0:
                    node01 = 0
                data1.append(node0)
                data2.append(node1)
                controlpower.append(node01)
                print(node)
            except:
                pass
    print("Finished......")
    data = [[data1,data2],controlpower]
    len_ = len(data1)
    len1 = int(len_*0.8)
    len2 = int(len_*0.9)
  
    train = [[data[0][0][:len1],data[0][1][:len1],data[0][2][:len1]],data[1][:len1]]
    valid = [[data[0][0][len1:len2],data[0][1][len1:len2],data[0][2][len1:len2]],data[1][len1:len2]]
    test = [[data[0][0][len2:],data[0][1][len2:],data[0][2][len2:]],data[1][len2:]]
  
    np.save('data/train.npy',train)
    np.save('data/valid.npy',valid)
    np.save('data/test.npy',test)
    return train,valid,test

if __name__ == '__main__':
    embedding_result = np.loadtxt('/embedding/EKGL.embed',dtype=float)
    train,valid,test = getdata(embedding_result)
    loss,val_loss,metrics,val_metrics = train_model([[train[0][0],train[0][1]],train[1]],[[valid[0][0],valid[0][1]],valid[1]],512)
    modelpath = 'model/CoNN.h5'
    