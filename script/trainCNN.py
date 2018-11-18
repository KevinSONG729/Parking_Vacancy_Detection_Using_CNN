# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 03:31:36 2018

@author: qs
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# TODO: add dropout layer

# save model file need
# !pip install h5py

import pickle
import os

basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..\\..\\')
[data, label] = pickle.load(open(os.path.join(basePath, 'dataset\\PKLotSegmented.pickle'), 'rb'))


# initialize model object
model = Sequential()

# construct the conv neural net
# first conv layer and pooling layer
model.add(Conv2D(256, (5, 5), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# second conv layer and pooling layer
model.add(Conv2D(256, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# this converts our 3D feature maps to 1D feature vectors
model.add(Flatten())

# fully connected layer
model.add(Dense(64))

# fully connected layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# start trianing with batch_size=1024 and 3 epochs and 30% of the dataset as testing set
model.fit(data, label, batch_size=1024, epochs=3, validation_split=0.3)

# save the model as a file
#model.save('model/PKLotCNN.model')
