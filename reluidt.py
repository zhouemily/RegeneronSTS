#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 03:16:03 2018

@author: Emily
"""

# USAGE
# python idt.py datasets modeltag activationfunction

# import the necessary packages
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from idtlib import datasets
from idtlib import models
import numpy as np
import argparse
import locale
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import plot_model
##3d drawing
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
#for mape
from sklearn.utils import check_array
from keras import losses

"""
arg[0]=dataset
arg[1]=tag: 8-4
arg[2]=act: relu
arg[3]=epoch: 50
arg[4]=batch: 16
"""
args=sys.argv[1:]
print(args)

tag=args[1]
act=args[2]
epochNum=int(args[3])
batch_sizeNum=int(args[4])
if not args[3]:
    epochNum=50
if not args[4]:
    batch_sizeNum=16

#set debug flag
DEBUG=False
#DEBUG=True
#current date and time
from datetime import datetime
now = datetime.now() # current date and time
tstamp = now.strftime("%m%d%Y%H%M%S")
if DEBUG:
   print("date and time:",tstamp)	

# construct the path to the input .txt file 
print("[INFO] loading idt attributes...")
if args[0]:
    inputPath = os.path.sep.join([args[0], "idt.txt"])
else:
    print("ERROR: invalid file path")
    sys.exit(1)

df = datasets.load_idt_attributes(inputPath,DEBUG)
df['Idt'] = pd.to_numeric(df['Idt'], errors='coerce')
if DEBUG:
    print("Idt column=")
    print(df['Idt'].head())

# construct a training and testing split with 75% of the data used
# for training and the remaining 25% for validation 
print("[INFO] constructing training/testing split...")
(train, test) = train_test_split(df, test_size=0.25, random_state=42)

# find the largest idt value in the training set and use it to
# scale our idt value to the range [0, 1] (this will lead to
# better training and convergence)
if DEBUG:
    print("training data:",train.head())
    print("testing data:",test.head())
maxIdt = train["Idt"].max()
if DEBUG:
    print("MaxIdt=",maxIdt)

trainY = train["Idt"] / maxIdt
testY  =  test["Idt"] / maxIdt

if DEBUG:
    print ("trainY.head:")
    print(trainY.head())
    print ("testY.head:")
    print(testY.head())

# process the idt attributes data by performing min-max scaling
# on continuous features
print("[INFO] processing data, normalization...")
(trainX, testX) = datasets.process_idt_attributes(df, train, test, DEBUG)

#C = np.delete(C, 1, 1)  # delete second column of C
#drop columns: "T", "Idt" from numpy array
trainX=np.delete(trainX,0,1)
testX=np.delete(testX,0,1)
trainX=np.delete(trainX,4,1)
testX=np.delete(testX,4,1)

# create  Mutiple layer Perceptron and then compile the model using mean absolute
# percentage error (MAPE or MAPD(deviation)) as the loss function, seek to minimize
# the absolute percentage difference between "Idt" *predictions*
# and the *actual Idt(sec)*
if not args[1] or not args[2]:
    print("ERROR: missing info for model creation")
    sys.exit(1)

model = models.create_mlp(DEBUG, trainX.shape[1], args[1], args[2],regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
#mape: 100/nSIG(yi-yt)/yt
##model.compile(optimizer=opt,loss='mean_squared_error',metrics=["mse"])
#mse: sqrt(SIG(yi-yt)^2/n)

model.summary()
plot_model(model, args[1]+args[2]+args[3]+args[4]+'_model.png', show_shapes=True)

#Train the model using tensorboard instance in the callbacks
# train the model
print("[INFO] training model...")
history=model.fit(trainX,trainY,validation_data=(testX, testY),epochs=epochNum, batch_size=batch_sizeNum,verbose=1)
preds = model.predict(trainX)

# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('idt_'+tag+"_"+act+args[3]+args[4]+"_"+tstamp+'.png')


print("[INFO] predicting Ignition Delay Time...")
preds = model.predict(testX)
#print("[INFO] mean_squared_error: testY,preds:")
#print(np.sqrt(mean_squared_error(testY,preds)))
if DEBUG:
    print("***********predicted normalized value****************")
    print(preds)
    print("************predicted inverted value***************")
    print("MaxIdt=",maxIdt)
    print(preds*maxIdt)
    print("***************************")

# compute the difference between the *predicted* idt value and the
# *actual* idt value, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
if DEBUG:
    print("=======pred - true=====")
    print(diff)
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
if DEBUG:
    print("=======(pred - true/tru)*100=====")
    print(absPercentDiff)
    print("=======(pred - true/tru)*100=====")

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. idt: {}, std idt: {}".format(df["Idt"].mean(), df["Idt"].std()))
print("[INFO] mean%: {:.2f}%, std%: {:.2f}%".format(mean, std))

#pred_train= model.predict(X_train)
#print(np.sqrt(mean_squared_error(y_train,pred_train)))
#pred= model.predict(X_test)
#print(np.sqrt(mean_squared_error(y_test,pred)))
#print(type(history.history['val_loss']))
tloss=history.history['loss']
vloss=history.history['val_loss']
#print("tloss len=",len(tloss))
#print("vloss len=",len(vloss))
print("================================")
print("mytopo=",args[1])
print("myact=",args[2])
print("myepoch=",args[3])
print("mybatch=",args[4])
print("myloss=",round(tloss[-1],2))
print("myval_loss=",round(vloss[-1],2))
