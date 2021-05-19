""" CDLNet/conv.py
Custom convolution modules for same-type convolution.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def adjoint(A):
	""" Adjoint operator for conv kernel
	"""
	return A.transpose(0,1).flip(2,3)

class Conv2d(nn.Module):
	""" Convolution with down-sampling.
	nic: num. input channels
	noc: num. output channels
	ks : kernel size (square kernels only)
	stride: down-sampling factor (decimation)
	"""
	def __init__(self, nic, noc, ks, stride=1):
		super(Conv2d, self).__init__()
		weight = torch.randn(noc,nic,ks,ks)
		weight = weight/torch.norm(weight, dim=(2,3), keepdim=True)
		self._weight = nn.Parameter(weight)
		self.stride  = stride
		p1 = int(np.floor((ks-1)/2))
		p2 = int(np.ceil((ks-1)/2))
		self._pad = (p1,p2,p1,p2)

	@property
	def weight(self):
		return self._weight.data
	@weight.setter
	def weight(self, data):
		self._weight.data = data

	def forward(self, x):
		W = self._weight
		pad_x = F.pad(x, self._pad, mode='constant')
		return F.conv2d(pad_x,W,stride=self.stride)

class ConvAdjoint2d(nn.Module):
	""" Convolution with zero-filling.
	nic: num. input channels
	noc: num. output channels
	ks : kernel size (square kernels only)
	stride: up-sampling factor (zero-filling)
	"""
	def __init__(self, nic, noc, ks, stride=2):
		super(ConvAdjoint2d, self).__init__()
		if stride < 2:
			raise ValueError("ConvAdjoint2d: stride should be >= 2.")
		weight = torch.randn(nic,noc,ks,ks)
		weight = weight/torch.norm(weight, dim=(2,3), keepdim=True)
		self._weight = nn.Parameter(weight)
		self.stride  = stride
		p1 = int(np.floor((ks-1)/2))
		p2 = int(np.ceil((ks-1)/2))
		self._pad = (p1,p2,p1,p2)
		self._output_padding = nn.ConvTranspose2d(1,1,ks,stride=self.stride)._output_padding

	@property
	def weight(self):
		return self._weight.data
	@weight.setter
	def weight(self, data):
		self._weight.data = data

	def forward(self, x):
		W = self._weight
		output_size = (x.shape[0], x.shape[1], self.stride*x.shape[2], self.stride*x.shape[3])
		op = self._output_padding(x, output_size,
		                          (self.stride, self.stride),
		                          (self._pad[0], self._pad[0]),
		                          (W.shape[3], W.shape[3]))
		return F.conv_transpose2d(x, W,
		                          padding = self._pad[0],
		                          stride  = self.stride,
		                          output_padding = op)

