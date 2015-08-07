__author__ = 'QQ'

import kNN_classification
group, labels = kNN_classification.createDataSet()

print kNN_classification.KNNClassification([0.9, 0.2, 0.4], group, labels, 3)
