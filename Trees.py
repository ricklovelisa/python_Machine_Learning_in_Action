# -*- coding: utf-8 -*-
__author__ = 'QQ'

# 决策树伪代码
# if so return 类标签;
# else
#   寻找划分数据集的最好特征
#   划分数据集
#   创建分支节点
#       for 每个划分的子集
#           调用函数createBranch并增加返回结果到分支节点中
#   return 分支节点

# 熵的计算
from math import log

def calcEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        Entropy -= prob * log(prob, 2)
    return Entropy

# 构建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]  # 这里用的是普通的数组
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        reducedFeatVec = featVec[:axis]
        reducedFeatVec.extend(featVec[axis + 1:])
        retDataSet.append(reducedFeatVec)
    return retDataSet
