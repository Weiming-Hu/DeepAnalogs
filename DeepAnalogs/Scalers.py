# "`-''-/").___..--''"`-._
#  (`6_ 6  )   `-.  (     ).`-.__.`)   WE ARE ...
#  (_Y_.)'  ._   )  `._ `. ``-..-'    PENN STATE!
#    _ ..`--'_..-_/  /--'_.' ,'
#  (il),-''  (li),'  ((!.-'
#
# Author: Weiming Hu <weiming@psu.edu>
#         Geoinformatics and Earth Observation Laboratory (http://geolab.psu.edu)
#         Department of Geography and Institute for CyberScience
#         The Pennsylvania State University
#
# This file includes definitions of scaler classes. I define this scaler classes manually
# so that they can be serialized for use in C++.
#
# References:
#
# https://pytorch.org/docs/stable/jit_language_reference.html#id2
#

import torch

        
@torch.jit.script
class StandardScaler:
    """
    This is a standardization scaler. It has been hard coded to standardize
    variables along the first dimension because this is designed to work for
    forecasts [parameters, stations, times, lead times]. Hard coding the dimensions
    makes it easier for PyTorch to serialize this class.

    Reference:
    https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/8
    """

    def __init__(self):
        self.mean = torch.zeros(0)
        self.std = torch.zeros(0)

    def fit(self, x):
        assert len(x.shape) == 4
        self.mean = x.mean((1, 2, 3), keepdim=True)
        self.std = x.std((1, 2, 3), unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-8)
        return x
    
    def reverse(self, x):
        x *= self.std
        x += self.mean
        return x
    
    def __str__(self):
        msg = "A torch Standardization scaler with\n" + \
              "- mean: shape {}\n".format(self.mean.shape) + \
              "- std:  shape {}".format(self.std.shape)
        
        return msg


@torch.jit.script
class MinMaxScaler:
    """
    This is a min-max scaler.
    
    Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    
    def __init__(self):
        self.min = torch.zeros(0)
        self.max = torch.zeros(0)
        self.diff = torch.zeros(0)
        
    def fit(self, x):
        assert len(x.shape) == 4
        num_features = x.size(0)
        self.min = torch.min(x.reshape(num_features, -1), dim=1).values
        self.max = torch.max(x.reshape(num_features, -1), dim=1).values
        
        self.min = self.min.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        self.max = self.max.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        self.diff = self.max - self.min
        
        if (self.diff == 0).sum() > 0:
            msg = 'The MinMax scaler forbids variables with the same value!\n Min: {}\n Max: {}'.format(
                self.min.squeeze(), self.max.squeeze())
            msg += '\nThese variables have zero variance: {}'.format(torch.nonzero(self.diff == 0))
            raise Exception(msg)
    
    def transform(self, x):
        x -= self.min
        x /= self.diff
        return x
    
    def reverse(self, x):
        x *= self.diff
        x += self.min
        return x
    
    def __str__(self):
        msg = "A torch MinMax scaler with\n" + \
              "- min: shape {}\n".format(self.min.shape) + \
              "- max: shape {}".format(self.max.shape)
        
        return msg
