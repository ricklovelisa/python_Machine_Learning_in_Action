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

