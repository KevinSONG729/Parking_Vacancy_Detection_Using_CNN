# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 00:19:17 2018

@author: Qiushi Li
"""

import tensorflow as tf
import cv2
#import os
#import pickle
#import random
from preprocessImage import processImage
#import time
#import pickle

#basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..\\..\\')
#[data, label] = pickle.load(open(os.path.join(basePath, 'dataset\\PKLotSegmented.pickle'), 'rb'))

model = tf.keras.models.load_model('model/PKLotCNN.model')

def predict(image):
    image = processImage(image).reshape([-1, 28, 28, 1])
    prediction = model.predict([image])

    if round(prediction[0][0]):
        #empty
        #print(prediction[0][0])
        return False
    else:
        #occupied
        #print(prediction[0][0])
        return True

if __name__ == '__main__':
    image = cv2.imread('test.jpg')
    if predict(image):
        print('Occupied')
    else:
        print('Empty')
