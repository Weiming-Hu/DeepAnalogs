#############################################################
# Author: Weiming Hu <weiminghu@ucsd.edu>                   #
#                                                           #
#         Center for Western Weather and Water Extremes     #
#         Scripps Institution of Oceanography               #
#         UC San Diego                                      #
#                                                           #
#         https://weiming-hu.github.io/                     #
#         https://cw3e.ucsd.edu/                            #
#                                                           #
# Date of Creation: 2022/02/22                              #
#############################################################
#
# This script implements the early stopping criteria.
#

import numpy as np
import torch.nn as nn


class EarlyStopping:
    """
    This class determines whether the training process should
    be early stopped based on current improvements.
    """
    
    def __init__(self, patience=None, verbose=True, delta=0.0):
        """
        Args:
            patience (int, optional): The number of epochs to wait before 
            termination if no improvements are detected. `None` to turn it off.
            Defaults to None.
            verbose (bool, optional): Whether to be verbose. Defaults to False.
            delta (float, optional): Minimum change in the loss to qualify 
            as an improvement. An improvement is detected when 
            `current_loss < recorded_best_loss + delta` is satisfied. Defaults to 0.
        """
        self.patience = np.Inf if patience is None else patience
        self.verbose = verbose
        self.delta = delta
        
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
    
    def __call__(self, loss):
        
        if loss < self.best_loss + self.delta:
            # An improvement is detected
            self.counter = 0
            self.best_loss = loss
            
        else:
            # No improvement is detected
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                
                if self.verbose:
                    msg = ('Maximum patience reached with no improvements after {} '
                           'epochs at best loss {}. Engage early stopping.'.format(
                               self.patience, self.best_loss))
                    
                    print(msg)
                    
        return self.early_stop
    