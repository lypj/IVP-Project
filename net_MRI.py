import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import conv, solvers, utils
from net import CDLNet
from utils import ifft2, fft2

class CDLNet_MRI(CDLNet):
    def __init__(self,
                 num_filters = 32,   # num. filters in each filter bank operation
                 num_inchans = 1,
                 filter_size = 7,    # square filter side length
                 stride = 1,         # strided convolutions
                 iters  = 10,         # num. unrollings
                 tau0   = 1e-2,      # initial threshold
                 adaptive = False,   # noise-adaptive thresholds
                 meansub = (2,3),    # mean subtraction dimensions 
                 init = True):
        super().__init__()

    def forward(self, y, sigma_n):
        
        c = sigma_n/255.0
        yp, params = utils.pre_process(y, self.stride)

        # LISTA
        z = ST(self.A[0](ifft2(yp).real), c*self.tau[0])
        
        for k in range(1, self.iters):
            z = ST(z - self.A[k]( ifft2( (fft2(self.B[k](z)) - yp) ).real ), c*self.tau[k])
            
        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat  = utils.post_process(xphat, params)
        return xhat.real, z
        

def ST(x,T):
    """ Soft-thresholding operation. 
    Input x, threshold T.
    """
    return x.sign()*F.relu(x.abs()-T)
