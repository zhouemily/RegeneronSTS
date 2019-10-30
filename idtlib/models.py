# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

def create_mlp(DEBUG, dim, tag, act, regress=False):
        # define our MLP network
        if DEBUG:
            print("dim="+str(dim),"tag="+tag,"act="+act)
        if not act:
            act="relu"
        model = Sequential()
        if tag=="8-4":
            model.add(Dense(8, input_dim=dim, activation=act))
            model.add(Dense(4, activation=act))
        elif tag=="64-16":
            model.add(Dense(64, input_dim=dim, activation=act))
            model.add(Dense(16, activation=act))
        elif tag=="128-64":
            model.add(Dense(128, input_dim=dim, activation=act))
            model.add(Dense(64, activation=act))
        else:
            model.add(Dense(20, input_dim=dim, activation=act))
            model.add(Dense(20, activation=act))
       

        # check to see if the regression node should be added
        if regress:
                model.add(Dense(1, activation="linear"))

        # return our model
        return model
