# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

def create_mlp(DEBUG, dim, tag, act, numa,regress=False):
        # define our MLP network
        num=numa
        if not num:
            num=16
        n=int(tag)
        if not n:
            print("input error: tag is null")
        if DEBUG:
            print("dim="+str(dim),"tag="+tag,"act="+act)
        if not act:
            act="relu"
        model = Sequential()
        model.add(Dense(num, input_dim=dim, activation=act))
        for i in range(n-1):
            model.add(Dense(num, activation=act))
       
        # check to see if the regression node should be added
        if regress:
                model.add(Dense(1, activation="linear"))

        # return our model
        return model
