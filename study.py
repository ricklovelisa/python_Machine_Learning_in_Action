# -*- coding: utf-8 -*-

# 简单knn分类学习
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
sortedClassCount2 = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)

# print  classCount.iteritems(), classCount.items(), sortedClassCount, sortedClassCount2
# 这个.items和.iteritems用法之间的区别还没有非常的清楚

# print dataSetSize, diffMat, sqDistances, distances, sortedDistIndicies, sortedDistIndicies[0], classCount.get()

# 绘图包的加载和使用
import matplotlib
import numpy
import scipy
import pyparsing
import matplotlib.pyplot as plt

# plt.plot([1,2,3])
# plt.ylabel('some numbers')
# plt.show()

# 归一化函数学习
# dataSet = array([[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 4.0, 2.0, 199.0], [30.0, 2.0, 2.0, 5.0, 3.0, 4.0, 5.0, 87.0, 10.0]])
# print kNN_classification.KNNClassification([0.9, 0.2, 0.4], group, labels, 3)

minVals = dataSet.min(0)
maxVals = dataSet.max(0)
ranges = maxVals - minVals
normDataSet = zeros(shape(dataSet))
m = dataSet.shape[0]
normDataSet = dataSet - tile(minVals, (m, 1))
normDataSet2 = normDataSet/tile(ranges, (m, 1))

print normDataSet, tile(ranges, (m, 1))