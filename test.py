# -*- coding: utf-8 -*-

from numpy import *
import Trees

myDat, labels = Trees.createDataSet()

print Trees.calcEntropy(myDat)
