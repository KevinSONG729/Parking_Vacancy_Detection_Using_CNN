# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 00:19:17 2018

@author: qs
"""

import tensorflow as tf
#import os
#import pickle
#import random
from preprocessImage import processImage
#import time
#import pickle

#basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..\\..\\')
#[data, label] = pickle.load(open(os.path.join(basePath, 'dataset\\PKLotSegmented.pickle'), 'rb'))

model = tf.keras.models.load_model('model/PKLotCNN.model')


image = processImage('test.jpg').reshape([-1, 28, 28, 1])
prediction = model.predict([image])

if round(prediction[0][0]):
    print('Empty Spot!')
else:
    print('Occupied Spot!')





    
    
    
    
    
    
    
    
    
    
    