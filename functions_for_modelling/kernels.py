import numpy as np
import sys


class uncorrelated_kernel():
    def __init__(self):
        self.efficient = True
        return

    def eval(self, x1, x2, data_size=None):
        if np.shape(x1) != np.shape(x2):
            sys.exit('Error in uncorrelated: Shapes not consistent. '
                     'Shapes are ' + str(np.shape(x1)) + ' and ' + str(np.shape(x2)))
        if np.array_equal(x1, x2):
            return 1
        return 0


class squared_exponential_kernel():
    def __init__(self, length):
        self.length = length
        self.efficient = True

    def eval(self, x1, x2, data_size=None):
        if np.shape(x1) != np.shape(x2):
            sys.exit('Error in fixed_range: Shapes not consistent. '
                     'Shapes are ' + str(np.shape(x1)) + ' and ' + str(np.shape(x2)))
        distance = x1 - x2
        squared_distance = distance ** 2
        sum_squared_distance = np.sum(squared_distance)
        scaled_sum_squared_distance = sum_squared_distance / (self.length ** 2)
        K = np.exp(-scaled_sum_squared_distance)
        return K
