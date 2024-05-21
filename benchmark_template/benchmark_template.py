import torch
import numpy as np

class BenchmarkFunction():

	def __init__(self,type = "discrete", **kwargs):
		self.type = type

	def initialize(self):
		pass

	def eval_noiseless(self,X):
		if X.size()[1] != self.d:
			raise AssertionError("Invalid dimension for the Benchmark function ...")
		pass

	def eval(self,X,sigma = 0.):
		z = self.eval_noiseless(X)
		y = z + sigma * torch.randn(X.size()[0], 1, dtype = torch.float64)
		return y

	def maximum_value(self):
		return 1.0

	def maximum(self, xtest = None):
		if self.type == "discrete":
			self.max = self.maximum_discrete(xtest)
		else:
			raise NotImplementedError()
		return self.max

	def maximum_discrete(self,xtest):
		maximum = torch.max(self.eval_noiseless(xtest))
		return maximum

