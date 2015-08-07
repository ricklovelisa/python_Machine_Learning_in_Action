__author__ = 'QQ'

from numpy import *

a = array([[2, 1], [1, 3], [10, 5]])
b = tile(a, (a.shape[0], 1))
c = b - 1

print range(8)