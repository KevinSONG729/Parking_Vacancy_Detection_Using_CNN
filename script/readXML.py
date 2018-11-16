# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:25:45 2018

@author: Qiushi Li
"""

import os
import xml.etree.ElementTree as et
import cv2
#from matplotlib import pyplot as plt
import numpy as np

GREEN = (0, 255, 0)
RED = (0, 0, 255)

# get current directory
basePath = os.path.dirname(os.path.realpath(__file__))
#TODO: loop through every xml file
xmlFilePath = os.path.join(basePath, 'dataset\\PKLot\\PKLot\\PKLot\\PUCPR\\Cloudy\\2012-09-12\\2012-09-12_06_20_57.xml')
# read xml file and get the tree
tree = et.parse(xmlFilePath)
parking = tree.getroot()

image = cv2.imread(os.path.join(basePath, 'dataset\\PKLot\\PKLot\\PKLot\\PUCPR\\Cloudy\\2012-09-12\\2012-09-12_06_20_57.jpg'))

for space in parking:
    spaceID = int(space.attrib['id'])
    occupied = bool(int(space.attrib['occupied']))
    for element in space:
        if element.tag == 'rotatedRect':
            for subelement in element:
                if subelement.tag == 'center':
                    center = (int(subelement.attrib['x']), int(subelement.attrib['y']))
                if subelement.tag == 'size':
                    size = (int(subelement.attrib['w']), int(subelement.attrib['h']))
                if subelement.tag == 'angle':
                    angle = int(subelement.attrib['d'])
        elif element.tag == 'contour':
            contour = []
            for subelement in element:
                contour.append([int(subelement.attrib['x']), int(subelement.attrib['y'])])
            contour = np.array(contour)
            if occupied:
                color = RED
            else:
                color = GREEN
            cv2.polylines(image, [contour], True, color, 3)
            
                
    
    #TODO: Get Training label
    #TODO: Get rid of the break

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()