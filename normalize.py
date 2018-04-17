import numpy as np

class Normalizer(object):
    def __init__(self):
        self.is_fit = False
        self.mean = []
        self.std = []
    
    def fit(self, matrix):
        try:
            self.std = matrix.std(axis=0)
            self.mean = matrix.mean(axis=0)
            self.max = matrix.max(axis=0)
            self.min = matrix.min(axis=0)
            self.is_fit = True
            return True
        except:
            return False

    def transform(self, matrix
        , move_mean_to_zero=False
        , scale_by_std=False
        , std_scale=2
        , scale_to_range=None):

        if not self.is_fit:
            raise Exception('model is not fit yet')

        matrix = matrix.copy()
        for i in range(len(self.std)):
            if move_mean_to_zero:
                matrix[:, i] -= self.mean[i]

            if scale_to_range is not None:
                a_min, a_max = scale_to_range
                matrix[:, i] = (matrix[:, i] - self.min[i]) \
                    / (self.max[i] - self.min[i]) \
                    * (a_max - a_min) \
                    + a_min

            elif scale_by_std:
                matrix[:, i] /= std_scale * self.std[i] 
            
        
        return matrix
