# -*- coding: utf-8 -*-


#Force google colab to switch to high ram mode. Run it only once. 
a = []
while(1):
    a.append('1')
#Also change runtime type to enable GPU from menu.

!pip install wfdb

##This block is only for access of files using google drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

import matplotlib.pyplot as plt
import numpy as np;
from random import shuffle;
from random import shuffle;
from tqdm import tqdm;
import wfdb;
import pickle 

import tensorflow;
from tensorflow.keras import layers;
from tensorflow.keras import Model;
from tensorflow.keras.optimizers import SGD;
from tensorflow.keras.callbacks import TensorBoard;

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
train_WFDB_file1 = drive.CreateFile({'id': '1xcBve5Pl3pCyemm9hA2YmnPyaTQaCyU-'});
train_WFDB_file2 = drive.CreateFile({'id': '1yqwtfMwq7EB6FMDmd7lYHZKpsIu2ngOz'});

#This block takes about 10 minutes to load the half of the training data.
train_WFDB_file1.GetContentFile('training_wfdb_Record1.rec');
train_WFDB_Rec = open('training_wfdb_Record1.rec', 'rb') 
train_WFDB_Rec = pickle.load(train_WFDB_Rec)

len(train_WFDB_Rec)

wfdb.plot_wfdb( record = train_WFDB_Rec[0], title='Record from Physionet Challenge 2020', figsize = (10, 10)) 
#wfdb documentation: https://wfdb.readthedocs.io/en/latest/index.html

display(train_WFDB_Rec[0].__dict__)

"""**From observation , only checksum and p-signal data is varying in diff recordings . So , I created two training set X for both the datas and y set.**

```
Label = ['AF','I-AVB','LBBB','Normal','PAC','PVC','RBBB','STD','STE']
The output will produce index corresponding to above list`
```
"""

import random
random.shuffle(train_WFDB_Rec) 

#CREATING TRAINING SET Y 
outputlist = ['AF','I-AVB','LBBB','Normal','PAC','PVC','RBBB','STD','STE']
train_set_y = np.zeros((len(train_WFDB_Rec) , 9))
for i in range(0,len(train_WFDB_Rec)):
  m = train_WFDB_Rec[i].__dict__['comments'][2]
  m = m.split(" ")[1]
  if (m in outputlist):
    index_no = outputlist.index(m)
    train_set_y[i][index_no] = 1
  else:
    l = m.split(",")[0]
    index_no = outputlist.index(l)
    train_set_y[i][index_no] = 1
    l = m.split(",")[1]
    index_no = outputlist.index(l)
    train_set_y[i][index_no] = 1

   
#CREATING TRAINING SET X (CHECKSUM)
train_set_x = np.zeros((len(train_WFDB_Rec), 12))
for i in range (0,len(train_WFDB_Rec)):
  train_set_x[i] = train_WFDB_Rec[i].__dict__['checksum']

#Creating train set x for p-signal

min_len = 5000
for i in range(0,len(train_WFDB_Rec)):
  length = len(train_WFDB_Rec[i].__dict__['p_signal'])
  if(length<min_len):
    min_len = length
print("min p-signal length " ,min_len , "\n but min length for 2nd record is 3000  so changed the set x dimensions")




train_set_X = np.zeros((len(train_WFDB_Rec), 3000,12))
for i in range(0,len(train_WFDB_Rec)):
  train_set_X[i] = np.copy(train_WFDB_Rec[i].__dict__['p_signal'][-3000:,:])

print("train_set_y :"  ,train_set_y.shape)
print("train_set_X :"  ,train_set_X.shape)
print("train_set_x :"  ,train_set_x.shape)

age = np.zeros((len(train_WFDB_Rec) , 1))
for i in range(0,len(train_WFDB_Rec)):
  m = train_WFDB_Rec[i].__dict__['comments'][0]
  m = m.split(" ")[1]
  age[i] = m

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf 
from tensorflow.keras.layers import Dense , Input , Activation, Flatten
from tensorflow.keras.models import Model

outputlist = ['AF','I-AVB','LBBB','Normal','PAC','PVC','RBBB','STD','STE']
occ = [0]*9
for i in range(len(train_set_y)):
    for j in range(0,8):
        if(train_set_y[i][j] == 1):
            occ[j] = occ[j]+1

"""#**Training on p-signal data**"""

train_x = train_set_X[:int(len(train_set_X)*0.8)]
train_y = train_set_y[:int(len(train_set_y)*0.8)]
test_x = train_set_X[int(len(train_set_X)*0.8):]
test_y = train_set_y[int(len(train_set_y)*0.8):]

len(train_y)

inputlayer = layers.Input(shape=(3000,12))       #creating input layer 

flatten_layer = Flatten()                     # instantiate the layer
x = flatten_layer(inputlayer) 
#x = Dense(10000, activation = tf.nn.relu)(x)   
x = Dense(1000, activation = tf.nn.relu)(x)   
#x = Dense(100, activation = tf.nn.relu)(x)                   
outputlayer= Dense(9, activation = tf.nn.softmax)(x)         #output layer 


model2 = Model(inputlayer, outputlayer);     #Training model from input layer and output layer 
model2.summary()

# Using 'SGD' optimizer and 'binary_crossentropy' loss



inputlayer = layers.Input(shape=(3000,12))       #creating input layer 

flatten_layer = Flatten()                     # instantiate the layer
x = flatten_layer(inputlayer) 
x = Dense(10000, activation = tf.nn.relu)(x)   
x = Dense(1000, activation = tf.nn.relu)(x)   
x = Dense(100, activation = tf.nn.relu)(x)                   
outputlayer= Dense(9, activation = tf.nn.softmax)(x)         #output layer 


model2 = Model(inputlayer, outputlayer);     #Training model from input layer and output layer 
model2.summary()
model2.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])   # SGD = schotastic gradient descent 
model2.fit(train_x,train_y, epochs = 70,validation_data=(test_x,test_y))    #epochs = no of rounds of training

"""GETTING RESULT"""

def compute_beta_score(labels, output, beta=2, num_classes=9):
    
    num_recordings = len(labels)

    fbeta_l = np.zeros(num_classes)
    gbeta_l = np.zeros(num_classes)
    fmeasure_l = np.zeros(num_classes)
    accuracy_l = np.zeros(num_classes)

    f_beta = 0
    g_beta = 0
    f_measure = 0
    accuracy = 0

    C_l=np.ones(num_classes);

    for j in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(num_recordings):
            
            num_labels = np.sum(labels[i])
        
            if labels[i][j] and output[i][j]:
                tp += 1/num_labels
            elif not labels[i][j] and output[i][j]:
                fp += 1/num_labels
            elif labels[i][j] and not output[i][j]:
                fn += 1/num_labels
            elif not labels[i][j] and not output[i][j]:
                tn += 1/num_labels

        if ((1+beta**2)*tp + (fn*beta**2) + fp):
            fbeta_l[j] = float((1+beta**2)* tp) / float(((1+beta**2)*tp) + (fn*beta**2) + fp)
        else:
            fbeta_l[j] = 1.0

        if (tp + fp + beta * fn):
            gbeta_l[j] = float(tp) / float(tp + fp + beta*fn)
        else:
            gbeta_l[j] = 1.0

        if tp + fp + fn + tn:
            accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
        else:
            accuracy_l[j] = 1.0

        if 2 * tp + fp + fn:
            fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            fmeasure_l[j] = 1.0


    for i in range(num_classes):
        f_beta += fbeta_l[i]*C_l[i]
        g_beta += gbeta_l[i]*C_l[i]
        f_measure += fmeasure_l[i]*C_l[i]
        accuracy += accuracy_l[i]*C_l[i]


    f_beta = float(f_beta)/float(num_classes)
    g_beta = float(g_beta)/float(num_classes)
    f_measure = float(f_measure)/float(num_classes)
    accuracy = float(accuracy)/float(num_classes)


    return accuracy,f_measure,f_beta,g_beta

pred_train_y = model2.predict(train_set_X)
pred_train_y.shape
print(train_set_y.shape , pred_train_y.shape)

#Rounding off predicted values to two decimal digit 
for i in range(0,len(pred_train_y)):
  for j in range(0,9):
    pred_train_y[i][j] = "%.3f" %pred_train_y[i][j]

    
#Creating two data frames with actual and predicted values 
import pandas as pd
actual = pd.DataFrame(train_set_y , columns = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE'])
predicted = pd.DataFrame(pred_train_y , columns = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE'])

#for any record ID , printing actual and predicted values 
index = 1000
print("Actual value :\n" , actual.iloc[[index]])
#print("\n")
print("Predicted value: \n" , predicted.iloc[[index]])

#testing set score
predicted = model2.predict(test_x)
import math
acc,_,f_b,g_b=compute_beta_score(test_y , predicted)
print('f2_score',f_b)
print('g2_score',g_b)
print('Geometric Mean',math.sqrt(f_b*g_b))

#Training set score
predicted = model2.predict(train_x)
import math
acc,_,f_b,g_b=compute_beta_score(train_y , predicted)
print('f2_score',f_b)
print('g2_score',g_b)
print('Geometric Mean',math.sqrt(f_b*g_b))

