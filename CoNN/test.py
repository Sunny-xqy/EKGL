import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from utils import *
import tensorflow as tf
from keras.models import load_model

import pickle
from utils import *
from walk import RWGraph

def read_enc():
    pkl_file = open('data/system_all.pkl','rb')
    enc =pickle.load(pkl_file)
    return enc

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

def predict(embedding_result):
    modelpath = 'model/CoNN.h5'
    model = load_model(modelpath,custom_objects={'myloss': myloss})
    file = open ('data/groundtruthforcontroller.txt', 'r')
    datas = file.readlines()
    file1 = open('data/predictresult.txt', 'w', encoding='utf-8')
    for data in datas:
        data1 = []
        data2 = []
        controlpower = []
        control = []
        try:
            links=data[:-1].split(';')
            enc = read_enc()
            for link in links[:-1]:
                node=link.split(',')
                node0 = embedding_result[enc.transform([node[0]])[0]]
                node1 = embedding_result[enc.transform([node[1]])[0]]
                node01= float(node[2])
                data1.append(node0)
                data2.append(node1)
                controlpower.append(node01)
                print(node)
                control.append(node)
        except:
            pass
        if controlpower != []:
            results = model.predict([data1,data2])
            print(results.size)
            # aa.append(acc)
            for ii in range(results.size):
                file1.write(str(control[ii][0])+','+str(control[ii][1])+','+str(results[ii][0])+';')
            file1.write('\n')
        else:
            file1.write('\n')

def test(data,y,model):
    result = model.predict(data)
    aa = []
    bb = []
    accuracy = 0
    tt = 0
    tf = 0
    ft = 0
    ff = 0
    errors = 0
    avg_y = sum(y)/len(y)
    for i in range(len(y)):
       
        aa.append((result[i]-y[i])**2)
        bb.append(abs(result[i]-y[i]))
        if abs(result[i]-y[i])<=0.01:
            accuracy = accuracy + 1
        if abs(result[i]-y[i])>=0.5:
            errors = errors + 1
        if result[i] >= 0.1 and y[i] >= 0.3:
            tt = tt + 1
        elif result[i] >= 0.1 and y[i] < 0.3:
            tf = tf + 1
        elif result[i] < 0.1 and y[i] >= 0.3:
            ft = ft + 1
        else:
            ff = ff + 1
    recall = tt/(tt+tf)
    precision = tt/(tt+ft) 
    f1 = 2*(tt)/(2*tt+ft+tf)
    print('Accuracy：{a}'.format(a=accuracy/len(y)))
    print('Recall：{a}'.format(a=recall))
    print('Precision：{a}'.format(a=precision))
    print('F1：{a}'.format(a = f1))
    print(sum(aa)/len(aa))
    print(sum(bb)/len(bb))
    return recall,precision,f1,sum(aa)/len(aa),sum(bb)/len(bb)


if __name__ == '__main__':
    embedding_result = np.loadtxt('/embedding/EKGL.embed',dtype=float)
    predict(embedding_result)