#!/usr/bin/env python3
"""
emily zhou
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
(trainX, testX) = datasets.process_idt_attributes(df, train, test,DEBUG)

# create our MLP and then compile the model using mean absolute
# percentage error as our loss, implying that we seek to minimize
# the absolute percentage difference between our price *predictions*
# and the *actual prices*
if not args[1] or not args[2]:
    print("ERROR: missing info for model creation")
    sys.exit(1)

model = models.create_mlp(DEBUG, trainX.shape[1], args[1], args[2],regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

model.summary()
plot_model(model, args[1]+args[2]+args[3]+args[4]+'_model.png', show_shapes=True)

##model.compile(optimizer=opt, loss='mean_squared_error',metrics=["mse"])

#Train the model using tensorboard instance in the callbacks
# train the model
print("[INFO] training model...")
history=model.fit(trainX,trainY,validation_data=(testX, testY),epochs=epochNum, batch_size=batch_sizeNum,verbose=1)

# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
"""
plt.plot(range(5))
plt.text(0.5, 3., "serif", family="serif")
plt.text(0.5, 2., "monospace", family="monospace")
plt.text(2.5, 2., "sans-serif", family="sans-serif")
plt.xlabel(r"µ is not $\mu$")
plt.tight_layout(.5)
plt.savefig("pgf_texsystem.pdf")
if DEBUG:
    plt.show()
"""
plt.savefig('idt_'+tag+"_"+act+args[3]+args[4]+"_"+tstamp+'.png')

print("[INFO] predicting Ignition Delay Time...")
preds = model.predict(testX)
print("***********predicted normalized value****************")
print(preds)
print("************predicted inverted value***************")
if DEBUG:
    print("MaxIdt=",maxIdt)
print(preds*maxIdt)
print("***************************")

# compute the difference between the *predicted* idt value and the
# *actual* idt value, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
print("=======pred - true=====")
print(diff)
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
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
