# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:25:45 2018

@author: Qiushi Li
"""
import cv2
import numpy as np
import os
import pickle
import random
import time

# TODO: build a class for loading the dataset, similar to mnist class
# download dataset
# unzip dataset
# process image
# number of data
# batch
# get train data
# get test data


# http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz


def processImage(image):
     
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    # divided by the maximum value to normalize each pixel
    gray = gray/np.max(gray)        
    # change to float 32
    gray = np.float32(gray)        
    # resize the image to 50 by 50 pixels
    gray = cv2.resize(gray, (28, 28))
    
    return gray

def prepareDataset(imageDir, batchSize=20000):
    dataset = []
    i = 0
    t0 = time.time()
    for root, dirs, files in os.walk(imageDir):
        # loop through every file in this folder and its subfolder
        for file in files:
            
            # get the path for the image file
            path = os.path.join(root, file)
            
            # read  the image file
            image = cv2.imread(path)
            
            # process image
            gray = processImage(image)
            
            # the folder name is the text label for this file
            textLabel = os.path.basename(os.path.dirname(path))        
            # if it's empty, label is [1, 0]
            # if it's occupied, label is [0, 1]
            label = 1 if textLabel=='Empty' else 0
            
            dataset += [(gray, label)]
            
            i += 1
            if i%batchSize == 0:
                timePerFile = round((time.time()-t0)/batchSize, 5)
                print('{}% finished with {}ms/image, ETA: {}sec.'.format(round(i/695899*100, 3), round(timePerFile*1000, 4), round((695899-i)*timePerFile, 3)))
                t0 = time.time()
    print('100% finished.')
            
    return dataset
        
    
if __name__ == "__main__":
    # set base path
    basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..\\..\\')
    
    # get the path for the segmented image folder
    imageDir = os.path.join(basePath, 'dataset\\PKLot\\PKLot\\PKLotSegmented\\')
    
    # get the dataset from the function
    # the dataset is a list which each element is a tuple
    # the first element of the tuple is the flattened image data
    # the second element of the tuple is the one-hot array label
    # we put the data in a list of tuple so that it won't be messed up after shuffle
    dataset = prepareDataset(imageDir)
    
    # shuffle the dataset
    random.shuffle(dataset)
    
    # extract the data and labels for the neural net
    data = np.array([x for (x, _) in dataset]).reshape(-1, 28, 28, 1)
    label = np.array([y for (_, y) in dataset])

    # save the data in a pickle file
    pickle.dump([data, label], open(os.path.join(basePath, 'dataset\\PKLotSegmented.pickle'), 'wb'))

