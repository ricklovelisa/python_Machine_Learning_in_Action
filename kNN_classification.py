# -*- coding: utf-8 -*-

from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1, 1.2], [1.1, 1.3, 1.0], [0, 0, 0], [0, 0.1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def KNNClassification(test, dataSet, labels, k):
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
    # operator.itemgetter(0), 定义了一个函数，其中key参数可以理解为，需要通过list或者字典中的哪一个域进行排序

    return sortedClassCount[0][0]
