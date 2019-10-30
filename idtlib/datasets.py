# import the necessary packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import glob
import os

def load_idt_attributes(inputPath,DEBUG):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	cols = ["T", "T1", "P", "E","D","Idt"]
	#cols = ["Tinverse", "Idt", "Pressure", "Dilution"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
	df['T'] = pd.to_numeric(df['T'], errors='coerce')
	df['T1'] = pd.to_numeric(df['T1'], errors='coerce')
	df['P'] = pd.to_numeric(df['P'], errors='coerce')
	df['E'] = pd.to_numeric(df['E'], errors='coerce')
	df['D'] = pd.to_numeric(df['D'], errors='coerce')
	df['Idt'] = pd.to_numeric(df['Idt'], errors='coerce')
	if DEBUG:
            print(df.head())
	# return the data frame
	return df

def process_idt_attributes(df, train, test, DEBUG):
	# initialize the column names of the continuous data
	cont = ["T", "T1", "P", "E","D","Idt"]

	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainX = cs.fit_transform(train[cont])
	testX = cs.transform(test[cont])
	if DEBUG:
            print((pd.DataFrame(trainX)).head())
            print((pd.DataFrame(testX)).head())

	# return the concatenated training and testing data
	return (trainX, testX)

def split_idt(X,y,test_size,random, DEBUG):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random)
    return (X_train, X_test, y_train, y_test)
