# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:05:05 2018

@author: qs
"""

from testCNN import predict
import cv2
import calibration
import numpy as np

GREEN = (0, 255, 0)
RED = (0, 0, 255)

surveillanceFeed1 = calibration.calibration('videoFeed/surveillanceFeed.mp4')
surveillanceFeed1.loadROI('roi/surveillanceFeed1.roi')

parkingSpot = surveillanceFeed1.parkingSpot

parkingSpot = parkingSpot[1:]

out = cv2.VideoWriter('output.mp4', -1, 20.0, (960,720))

while True:
    _, frame = surveillanceFeed1.video.read()
    rows, cols, ch = frame.shape
    for spot in parkingSpot:

        # get a tilted ractangle box
        rect = cv2.minAreaRect(spot)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # get box width and height
        boxHeight = int(round(np.linalg.norm(box[0]  - box[1])))
        boxWidth = int(round(np.linalg.norm(box[1] - box[2])))
        #cv2.drawContours(frame,[box],0,(0,0,255),2)

        # calculate the angle of the box
        vec1 = box[1] - box[0]
        vec2 = np.array([-1, 0])
        theta = np.arccos(vec1.dot(vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2)))/np.pi*180

        # rotate the image
        M = cv2.getRotationMatrix2D((box[2][0], box[2][1]), theta, 1)
        rotatedImage = cv2.warpAffine(frame, M, (cols, rows))

        roi = rotatedImage[box[2][1]:boxWidth+box[2][1], box[2][0]:box[2][0]+boxHeight]
        
        if predict(roi):
            color = RED
        else:
            color = GREEN
        cv2.polylines(frame, [spot], True, color, 1)

    #out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
