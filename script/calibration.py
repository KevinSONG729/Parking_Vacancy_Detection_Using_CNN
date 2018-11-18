# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:13:50 2018

@author: Qiushi Li
"""

import cv2
import numpy as np
import pickle

class calibration:
    def __init__(self, videoPath):
        # load the video
        self.loadVideo(videoPath)
        
        # get the first frame
        _, self.frame = self.video.read()
        # we will mark the self.marked image instead of the self.frame
        self.marked = np.copy(self.frame)
        
        # initialize variable
        self.parkingSpot = []
        self.contour = []
        
    
    def loadVideo(self, videoPath):
        self.video = cv2.VideoCapture(videoPath)
    
    def markup(self, event,x,y,flags,param):
        # if left mouse button pressed
        if event == cv2.EVENT_LBUTTONDOWN:
            # get the mouse position
            self.contour.append([x, y])
            # if there are 4 points
            if len(self.contour) == 4:
                # convert to np array
                self.contour = np.array(self.contour)
                # save the mark in self.parkingSpot
                self.parkingSpot.append(self.contour)
                # mark the image
                cv2.polylines(self.marked, [self.contour], True, (255, 255, 255), 1)
                # clear the self.contour variable for next mark
                self.contour = []
    
    def saveROI(self, path):
        pickle.dump(self.parkingSpot, open(path, 'wb'))
        
    def loadROI(self, path):
        # load from pickle
        self.parkingSpot = pickle.load(open(path,'rb'))
        
        # mark on self.marked
        for contour in self.parkingSpot:
            cv2.polylines(self.marked, [contour], True, (255, 255, 255), 1)
        
    def showFrame(self):
        cv2.imshow('Frame', self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def showMarked(self):
        cv2.imshow('Marked', self.marked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def startMarkup(self):
        self.parkingSpot = []
        cv2.namedWindow('marked')
        cv2.setMouseCallback('marked',self.markup)
        
        while True:
            cv2.imshow('marked',self.marked)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()


#if __name__=='__main__':
#    c = calibration('')




