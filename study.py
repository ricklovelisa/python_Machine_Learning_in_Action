# -*- coding: utf-8 -*-

from numpy import *
import kNN_classification
import operator

a = array([[2, 1], [1, 3], [10, 5]])
b = tile(a, (a.shape[0], 1))
c = b - 1

dataSet, labels = kNN_classification.createDataSet()
test = [0.9, 0.2, 0.4]
k = 3

dataSetSize = dataSet.shape[0]
diffMat = tile(test, (dataSetSize, 1)) - dataSet
sqDiffMat = diffMat ** 2
sqDistances = sqDiffMat.sum(axis = 1)
distances = sqDistances ** 0.5
sortedDistIndicies = distances.argsort()
classCount = {}
for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)

print  classCount, sorted(classCount), sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
# print dataSetSize, diffMat, sqDistances, distances, sortedDistIndicies, sortedDistIndicies[0], classCount.get()