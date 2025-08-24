from mutedpy.benchmark_template.protein_operator import ProteinOperator
from mutedpy.benchmark_template.benchmark_template import BenchmarkFunction
import pandas as pd
import numpy as np
import torch

class ProteinBenchmark(BenchmarkFunction):

	def __init__(self, **kwargs):

		"""
		initialize the protein benchmark_template

		 fname : dataset name
		 dim : dimension of the dataset
		 ref : for smaller dimensions what is the reference in the 4 dim space?
		 avg : average the effect over other combinations in lower dimensions
		"""
		"""
		Convention of the following dictionary is to map B->D as B can stand for N and D.
		"""
		self.load()

	def load(self, **kwargs):
		pass

	def get_real_name(self, name):
		out = []
		for i in name:
			out.append(self.real_names[i])
		return out

	def translate(self,X):
		f = lambda x: self.dictionary[x]
		Y = X.copy()
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				Y[i,j] = f(X[i,j])
		return Y.astype(int)

	def translate_one_hot(self,X):
		try:
			Y = self.translate(X)
		except:
			Y = X
		n,d = list(X.shape)
		Z = np.zeros(shape=(n,d*self.total))
		for i in range(n):
			for j in range(d):
				Z[i,Y[i,j]+j*self.total] = 1.0

		return Z

	def self_translate(self):
		"""
		self translate from
		:return:
		"""
		f = lambda x: self.dictionary[x]
		for j in range(4):
			self.data['P'+str(j+1)] = self.data['P'+str(j+1)].apply(f)

	def set_fidelity(self,F):
		self.Fidelity = F

	def scale(self):
		self.scale = 1

	def eval_noiseless(self,X):
		"""
		evaluate depends on the dimension
		"""
		n, _ = X.size()
		res = torch.random.randn(size = (n,1)).double()
		return res

	def interval_number(self):
		arr = self.interval_letters()
		out = self.translate(arr)
		return out

	def interval_onehot(self):
		arr = self.interval_letters()
		out = self.translate_one_hot(arr)
		return out

	def interval_letters(self):
		names = list(self.dictionary.keys())
		arr = []
		for i in range(self.dim):
			arr.append(names)
		out = helper.cartesian(arr)
		return out

	def subsample_dts(self, N, split = 0.90):
		xtest = self.interval_onehot()
		(n,d) = xtest.shape
		sample = xtest[np.random.randint(0,n,N),:]
		y_sample = self.eval_one_hot(sample)

		x_train = sample[0:int(np.round(split*N)),:]
		y_train = y_sample[0:int(np.round(split*N)),:]
		x_test = sample[int(np.round(split*N)):N,:]
		y_test = sample[int(np.round(split*N)):N,:]

		return (x_train,y_train,x_test,y_test)


	def eval_fidelity(self,X):
		return self.Fidelity(X)

	def eval(self,X):
		z = self.eval_noiseless(X)
		return z

	def eval_one_hot(self,X):
		n, d = list(X.shape)
		Z = np.zeros(shape=(n, self.dim ))
		for i in range(n):
			for j in range(d):
				if 	X[i,  j] > 0:
					Z[i,j // self.total] = j % self.total
		Z = Z.astype(int)
		Y = self.eval(Z)
		return Y

	def plot_one_site_map(self,kernel):
		names = list(self.dictionary.keys())
		xtest = self.data['P1'].values.reshape(-1, 1)
		xtest = self.translate_one_hot(xtest)
		xtest = torch.from_numpy(xtest)
		ax = plt.imshow(kernel(xtest, xtest))
		plt.colorbar()
		real_names = self.get_real_name(names)
		plt.xticks(range(xtest.shape[0]),real_names,fontsize=18, rotation= 60)
		plt.yticks(range(xtest.shape[0]),real_names,fontsize=18)
		plt.margins(0.2)
		plt.show()

if __name__ == "__main__":
	Benchmark = ProteinBenchmark("protein_data_gb1.h5", dim = 2, ref = ['A','B','C','D'])
	#print (Benchmark.data)
	Benchmark.self_translate()
	Benchmark.data.plot.scatter(x='P1', y='P2', c=Benchmark.data['Fitness'], s = 200)
	#print (Benchmark.data)
	X = np.array([['F','C'],['D','C']])
	X_ = Benchmark.translate(X)
	print (X,X_)
	X__ = Benchmark.translate_one_hot(X)

	print (Benchmark.translate_one_hot(X))

	print (Benchmark.eval(X_))

	print (Benchmark.eval_one_hot(X__))
