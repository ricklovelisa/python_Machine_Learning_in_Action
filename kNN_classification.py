# -*- coding: utf-8 -*-

from numpy import *
import operator

# 简单knn算法训练集（确切说不能叫训练集）
def createDataSet():
    group = array([[1.0, 1.1, 1.2], [1.1, 1.3, 1.0], [0, 0, 0], [0, 0.1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 简单KNN分类实现
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

# 将文件里的数据输入为python矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines, 3))       # 定义一个0矩阵，行数为numberOfLines，列数为3
    classLbelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()                     # 去除前后空格
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLbelVector.append(int(listFromLine[-1]))
        index += 1
    return  returnMat, classLbelVector

# 归一化数据至（0，1）之间, 这个方法对于数据中的最小值来说结果为0，最大值，结果为1。
def autoNorm(dataSet):
    minVals = dataSet.min(0)                    # 参数0可以让函数从列中选取最小值或最大值，而不是当前行
    maxVals = dataSet.max(0)                    # 例如在[[1, 2, 3], [4, 0, 10]]的矩阵中，.min(0)得到的结果是[1， 0, 3]，
    ranges = maxVals - minVals                  # min(1)得到的结果是[1，0]
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


