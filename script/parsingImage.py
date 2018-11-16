# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:25:45 2018

@author: Qiushi Li
"""
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import random
import operator

t0 = time.time()
# get current directory
basePath = os.path.dirname(os.path.realpath(__file__))

image = cv2.imread(os.path.join(basePath, '..\\..\\dataset\\PKLot\\PKLot\\PKLot\\PUCPR\\Cloudy\\2012-09-12\\2012-09-12_06_20_57.jpg'))


# preprocess image
def preprocessImage(image):
    # denoise the image
    denoised = cv2.bilateralFilter(image,9,50,50)

    # convert to grayscale
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # calculate the canny edge detector parameter
    v = np.median(gray)
    sigma = 0.33
    #---- apply optimal Canny edge detection using the computed median----
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))

    # do the Canny edge detection
    edges = cv2.Canny(gray,lower_thresh,upper_thresh)

    return edges

edges = preprocessImage(image)
# get the coordinates of each edge point
points = []
countx = np.shape(edges)[0]
for row in edges:
    county = 0
    for element in row:
        if element:
            points.append([county, countx, 1])
        county += 1
    countx -= 1

points = np.array(points)


def crossProduct(a, b):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]).dot(b)

def getProposal(points, numOfProposal):
    pointNum = random.sample(range(len(points)), 2*numOfProposal)
    lineParameters = []
    for i in range(0, len(pointNum), 2):
        point1 = points[pointNum[i], :]
        point2 = points[pointNum[i+1], :]
        lineParameters.append(crossProduct(point1, point2))
    return np.array(lineParameters)

def getBestProposal(proposals, points, sigmaGM):
    # get weighted error from every point and add together
    weightedError = []
    for proposal in proposals:
        rho = 0
        for point in points:
            error = proposal.dot(point)
            # use GM-estimator to weight the error
            rho += error**2 / (sigmaGM**2 + error**2)
        weightedError.append(rho)
    weightedError = np.array(weightedError)
    min_index, min_value = min(enumerate(weightedError), key=operator.itemgetter(1))
    return proposals[min_index, :]

def fitLineIRLS(points, initialLine, sigmaGM):
    U = np.vstack((points[:, 0], np.ones((1, len(points))))).T
    V = points[:, 1]

    error = np.zeros((1, len(points)))[0]
    weight = np.zeros((1, len(points)))[0]
    counter = 0
    while True:
        i = 0
        for point in points:
            error[i] = bestProposal.dot(point)
            weight[i] = 2 * sigmaGM**2 / (error[i]**2 + sigmaGM**2)**2
            i += 1

        W = np.diag(weight.T)
        parameter = np.linalg.inv(U.T.dot(W).dot(U)).dot(U.T.dot(W).dot(V))
        parameter = [-parameter[0], 1, -parameter[1]]
        if abs(np.linalg.norm(parameter) - np.linalg.norm(initialLine)) < 0.01:
            return parameter
        initialLine = parameter
        if counter > 100:
            print('IRLS failed, parameter won\'t converge.')
            break

sigmaGM = 1
proposals = getProposal(points, 100)
bestProposal = getBestProposal(proposals, points, sigmaGM)
bestProposal = fitLineIRLS(points, bestProposal, sigmaGM)
# if parameter of y is not zero (not vertical line)
if bestProposal[1]:
    x = np.array([0, 1300])
    y = -bestProposal[0]/bestProposal[1] * x + (-bestProposal[2]/bestProposal[1])
else:
    y = [0, 700]
    x = [-bestProposal[2]/bestProposal[0], -bestProposal[2]/bestProposal[0]]
print(time.time() - t0)
plt.figure(figsize=(6, 4))
plt.scatter(points[:,0], points[:,1], s=0.01)
plt.plot(x,y, linewidth=1, c='red')
plt.xlim((0, 1300))
plt.ylim((0, 700))
plt.figure()
plt.imshow(edges)
plt.show()

#cv2.waitKey(0)
#cv2.destroyAllWindows()
