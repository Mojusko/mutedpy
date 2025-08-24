import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import pickle
import numpy as np

# loaders

# utils

# scores

# processing

# models
from mutedpy.protein_learning.embeddings.amino_acid_embedding import AminoAcidEmbedding
from stpy.kernel import KernelFunction
from stpy.regression.gauss_procc import GaussianProcess
from stpy.embeddings.polynomial_embedding import CustomEmbedding
from stpy.embeddings.onehot_embedding import OnehotEmbedding
from pymanopt.manifolds import Euclidean
from mutedpy.protein_learning.regression.regression_basis import ProteinKernelLearner
from stpy.helpers.helper import full_group

class ARDModelLearner(ProteinKernelLearner):

	def __str__(self):
		return "ard_model"

	def preload(self):
		if self.feature_mask is not None:
			self.Embedding = self.feature_loader
			self.Embedding_original = self.Embedding
		else:
			self.Embedding = self.feature_loader

	def load(self):
		if not self.loaded:
			self.preload()

			# all phis to do model selection?
			if self.model_selection_all == True:
				phi = self.Embedding.embed(self.x)
				self.feature_selector.pass_data(phi, self.y)
			else:
				phi = self.Embedding.embed(self.x_train)
				self.feature_selector.pass_data(phi, self.y_train)
			self.feature_mask = self.feature_selector.select(self.topk)

			self.Embedding_original = self.Embedding
			embed = lambda x: self.Embedding_original.embed(x)[:, self.feature_mask]
			d = self.feature_mask.size()[0]

			self.Embedding = CustomEmbedding(5, embed, d)
			print("Final feature dim:", d, "from requested", self.topk)
			self.feature_names = self.Embedding_original.feature_names
			self.loaded = True

	def add_data_points(self,phinext,ynext):
		self.GP.add_data_point(phinext,ynext)

	def just_fit(self):
		phi_train = self.Embedding.embed(self.x_train)
		self.GP.fit_gp(phi_train, self.y_train)

	def embed(self, x):
		phi_test = self.Embedding.embed(x)
		return phi_test

	def save_model(self, id = "", save_loc = None, special_identifier = ''):
		d = super().save_model(id = id, save_loc = save_loc, special_identifier = special_identifier)
		# selected features
		d['feature_mask'] = self.feature_mask
		d['feature_names'] = self.feature_names
		# what kernel
		d['kernel'] = self.ard_kernel
		# gamma
		d['kernel_object'] = self.GP.kernel_object
		d['ard_gamma'] = self.GP.kernel_object.ard_gamma
		# noise
		d['loss'] = self.loss
		d['noise_std'] = self.GP.s
		d['x'] = self.GP.x
		d['y'] = self.GP.y
		d['feature_loader'] = self.feature_loader

		if save_loc is None:
			filename = self.results_folder + "/model_params_" + id + special_identifier + ".p"
		else:
			filename = save_loc

		with open(filename, "wb") as f:
			pickle.dump(d, f)

	def fit(self, optimize=True, save_loc = None):
		self.load()
		phi_train = self.Embedding.embed(self.x_train)
		d = self.Embedding.m
		print ("Feature size:",d)

		if self.proj_dim is None:
			self.proj_dim = d

		print (self.additive)
		if self.additive:
			groups = full_group(d)
			print ("Creating fully additive model.")
			print (groups)
		else:
			groups = None

		k = KernelFunction(kernel_name=self.ard_kernel, kappa=3.,
						   ard_gamma=torch.ones(d).double() * 0.01, groups = groups,
						   d=d, nu = 2.5, cov = torch.randn(d,self.proj_dim).double())
		print (k.description())
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s=self.s, loss = self.loss)

		print ("Data size:",phi_train.size()[0])
		self.GP.fit_gp(phi_train, self.y_train)

		if save_loc is None:
			save_loc = self.results_folder + "/model_" + str(self.split) + ".np"

		if phi_train.size()[0] > 3000:
			ind = np.random.choice(np.arange(0, phi_train.size()[0], 1), 3000)
			print (ind)
			subselection_phi = phi_train[ind, :]
			subselection_y = self.y_train[ind, :]
			self.GP.fit_gp(subselection_phi, subselection_y)

		if optimize:
			if self.ard_kernel != "full_covariance_se" and  self.ard_kernel != "full_covariance_matern":
				self.GP.optimize_params(type="bandwidth", restarts=self.restarts, verbose=2, maxiter=self.maxiter,
									mingradnorm=10e-5, optimizer='pytorch-minimize', scale=10., save=True,
									init_func=self.init_func,save_name= save_loc)
				print(self.GP.kernel_object.description())
			else:
				init_func = lambda x: torch.randn(d,self.proj_dim).double() * 1.
				self.GP.optimize_params_general(
					params={'0': {"cov": (init_func, Euclidean(d * self.proj_dim), None)}},
					restarts=self.restarts, verbose=True, maxiter=self.maxiter, mingradnorm=10e-5,
					optimizer='pytorch-minimize',save_name=save_loc)

				print("OPTIMIZED COV OBJECT:")
				print(self.GP.kernel_object.cov.size())

		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True

	def predict(self, x = None):
		if x is None:
			x = self.x_test
		else:
			pass

		self.mu_train, self.std_train = self.GP.mean_std(self.Embedding.embed(self.x_train))
		#stepby = 20000
		#print(x.size()[0] % stepby)
		#if x.size()[0]<stepby:
		self.mu, self.std = self.GP.mean_std(self.Embedding.embed(x))
		return self.mu, self.std
		# else:
		# 	x = self.Embedding.embed(x)
		# 	self.mu = torch.zeros(size = (x.size()[0],1)).double()
		# 	self.std = torch.zeros(size=(x.size()[0],1)).double()
		# 	for i in np.arange(0,x.size()[0]//stepby,1):
		# 		print (i,"/",x.size()[0]//stepby)
		# 		self.mu[i*stepby:(i+1)*stepby],self.std[i*stepby:(i+1)*stepby] = self.GP.mean_std(x[i*stepby:(i+1)*stepby,:], reuse=True)
		# 	if x.size()[0]%stepby>0:
		# 		self.mu[x.size()[0]-x.size()[0]%stepby:],self.std[x.size()[0]-x.size()[0]%stepby:] = self.GP.mean_std(x[x.size()[0]-x.size()[0]%stepby:,:], reuse = True)
		# 	return self.mu, self.std

class ARDLinearModelLearning(ARDModelLearner):
	def __str__(self):
		return "ard_model+linear"

	def fit(self):
		self.Embedding = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.Embedding.load_projection("data/projection-dim5-norm.pt")
		phi_train = self.Embedding.embed(self.x_train)
		k = KernelFunction(kernel_name = "ard",kappa = 0.5, ard_gamma = torch.ones(self.Embedding.projected_components).double()*0.01, d = self.Embedding.projected_components)
		kernel_function2 = lambda x, y, kappa, group: (x @ y.T).T + 1.
		k2 = KernelFunction(kernel_function=kernel_function2, d=self.Embedding.projected_components)
		k = k + k2

		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s, loss = self.loss)
		self.GP.fit_gp(phi_train,self.y_train)
		self.GP.optimize_params(type="bandwidth", restarts=self.restarts, verbose = True, maxiter=self.maxiter, mingradnorm = 10e-5, optimizer = 'pytorch-minimize', scale = 10)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True

class AdditiveARDModelLearning(ARDModelLearner):
	def __str__(self):
		return "ard_model_additive"

	def load(self):

		self.Embedding = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.Embedding.project_individual_pca(5)
		self.Embedding.set_projection()
		self.groups = [[19 * j + i for i in range(19)] for j in range(5)]

	def fit(self):
		self.load()
		d = self.Embedding.m
		phi_train = self.Embedding.embed(self.x_train)
		k = KernelFunction(kernel_name = "ard",kappa = 3., ard_gamma = torch.ones(d).double()*0.01,
			d = d, groups=self.groups)
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s)
		self.GP.fit_gp(phi_train,self.y_train)
		self.GP.optimize_params(type="bandwidth", restarts=self.restarts, verbose = True, maxiter=self.maxiter,
								mingradnorm = 10e-5, optimizer = 'pytorch-minimize', scale = 10)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True


class PairwiseRoundARDModelLearning(ARDModelLearner):
	def __str__(self):
		return "ard_parwise_round"

	def fit(self):

		self.Embedding = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.Embedding.project_individual_pca(5)
		self.Embedding.set_projection()

		groups = [ [(19*j+i)%95 for i in range(19*2)] for j in range(5)]

		phi_train = self.Embedding.embed(self.x_train)
		k = KernelFunction(kernel_name = "ard",kappa = 3., ard_gamma = torch.ones(95).double()*0.01,
			d = 95, groups=groups)
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s)
		self.GP.fit_gp(phi_train,self.y_train)
		self.GP.optimize_params(type="bandwidth", restarts=self.restarts, verbose = True, maxiter=self.maxiter, mingradnorm = 10e-5, optimizer = 'pytorch-minimize', scale = 10)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True


class PairwiseAllARDModelLearning(ARDModelLearner):
	def __str__(self):
		return "ard_parwise_all"

	def load(self):
		self.Embedding = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.Embedding.project_individual_pca(5)
		self.Embedding.set_projection()

	def fit(self):
		self.load()
		m = self.Embedding.m
		gg = [ [(19*j+i)%95 for i in range(19)] for j in range(5)]
		groups = []
		for g1 in gg:
			for g2 in gg:
				if g1 != g2:
					groups.append(g1+g2)
		no_groups = len(groups)
		phi_train = self.Embedding.embed(self.x_train)
		k = KernelFunction(kernel_name = "ard",kappa = 3., ard_gamma = torch.ones(m).double()*0.01,
			d = m, groups=groups)
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s)
		self.GP.fit_gp(phi_train,self.y_train)
		self.GP.optimize_params(type="bandwidth", restarts=self.restarts,
								verbose = True, maxiter=self.maxiter,
								mingradnorm = 10e-5, optimizer = 'pytorch-minimize', scale = 100)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True


class PairwiseAllARDModelLearningGeometric(ARDModelLearner):
	def __str__(self):
		return "ard_parwise_all_Geometric"

	def load(self):
		self.EmbeddingA = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.EmbeddingA.project_individual_pca(5)
		self.EmbeddingA.set_projection()
		self.EmbeddingA.normalize_features(n_sites=5)

		self.EmbeddingG = ContactMap("data/new_features.csv", truncation=False)
		self.EmbeddingG.normalize(xtest=True)
		self.EmbeddingG.restrict_to_varaint(std=0.0001)

		self.embed = lambda x: torch.hstack([self.EmbeddingA.embed(x), self.EmbeddingG.embed(x)])
		self.Embedding = CustomEmbedding(5,self.embed,self.EmbeddingA.m+self.EmbeddingA.m)


	def fit(self):
		self.load()
		m = self.Embedding.m
		gg = [ [(19*j+i)%95 for i in range(19)] for j in range(5)]
		groups = []
		for g1 in gg:
			for g2 in gg:
				if g1 != g2:
					groups.append(g1+g2)

		groups = groups + [[i+19*5 for i in range(self.EmbeddingG.m)]]
		no_groups = len(groups)
		phi_train = self.Embedding.embed(self.x_train)
		k = KernelFunction(kernel_name = "ard",kappa = 3., ard_gamma = torch.ones(m).double()*0.01,
			d = m, groups=groups)
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s)
		self.GP.fit_gp(phi_train,self.y_train)
		self.GP.optimize_params(type="bandwidth", restarts=self.restarts,
								verbose = True, maxiter=self.maxiter,
								mingradnorm = 10e-5, optimizer = 'pytorch-minimize', scale = 100)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True



class SquaredExpGroupModelLearning(ARDModelLearner):
	def __str__(self):
		return "se_group"

	def fit(self):

		self.Embedding = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.Embedding.project_individual_pca(5)
		self.Embedding.set_projection()

		gg = [ [(19*j+i)%95 for i in range(19)] for j in range(5)]
		groups = []
		for g1 in gg:
			for g2 in gg:
				if g1 != g2:
					groups.append(g1+g2)
		no_groups = len(groups)
		phi_train = self.Embedding.embed(self.x_train)
		k = KernelFunction(kernel_name = "squared_exponential_per_group",params={'gamma_per_group':torch.ones(no_groups).double()*0.01},  ard_gamma = torch.ones(95).double()*0.01,
		d = 95, groups=groups, kappa = 3.)
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s)
		self.GP.fit_gp(phi_train,self.y_train)
		self.GP.optimize_params_general(params={'0':{"gamma_per_group":(torch.ones(no_groups).double()*0.01,Euclidean(no_groups),None)}},
	restarts=self.restarts, verbose = True, maxiter=self.maxiter, mingradnorm = 10e-5, optimizer = 'pytorch-minimize', scale = 100)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True

class ARDPerGroupModelLearning(ARDModelLearner):
	def __str__(self):
		return "ard_per_group_group"

	def fit(self):

		self.Embedding = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.Embedding.project_individual_pca(5)
		self.Embedding.set_projection()

		gg = [ [(19*j+i)%95 for i in range(19)] for j in range(5)]
		groups = []
		for g1 in gg:
			for g2 in gg:
				if g1 != g2:
					groups.append(g1+g2)
		no_groups = len(groups)
		phi_train = self.Embedding.embed(self.x_train)
		k = KernelFunction(kernel_name = "ard_per_group",kappa = 3.,params={'ard_per_group':torch.ones(no_groups*19*2).double()*0.01},
  ard_gamma = torch.ones(95).double()*0.01, d = 95, groups=groups)
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s)
		self.GP.fit_gp(phi_train,self.y_train)
		self.GP.optimize_params_general(params={'0':{"ard_per_group":(torch.ones(no_groups*19*2).double()*0.01,Euclidean(no_groups*19*2),None)}},
	 restarts=self.restarts, verbose = True, maxiter=self.maxiter, mingradnorm = 10e-5, optimizer = 'pytorch-minimize', scale = 10)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True


class ARDPerGroupModelLearningOnehot(ARDModelLearner):
	def __str__(self):
		return "onehot_ard_per_group_group"

	def fit(self):

		self.Embedding = OnehotEmbedding(20,5)

		gg = [ [(20*j+i)%100 for i in range(20)] for j in range(5)]
		groups = []
		for g1 in gg:
			for g2 in gg:
				if g1 != g2:
					groups.append(g1+g2)
		no_groups = len(groups)

		phi_train = self.Embedding.embed(self.x_train)
		k = KernelFunction(kernel_name = "ard_per_group",kappa = 3.,params={'ard_per_group':torch.ones(no_groups*20*2).double()*0.01},
  ard_gamma = torch.ones(100).double()*0.01, d = 95, groups=groups)
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s)
		self.GP.fit_gp(phi_train,self.y_train)
		self.GP.optimize_params_general(params={'0':{"ard_per_group":(torch.ones(no_groups*20*2).double()*0.01,Euclidean(no_groups*20*2),None)}},
	 restarts=self.restarts, verbose = True, maxiter=self.maxiter, mingradnorm = 10e-5, optimizer = 'pytorch-minimize', scale = 10)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True

class ARDModelLearningOnehot(ARDModelLearner):

	def __str__(self):
		return "onehot_ard_model"

	def fit(self):
		self.Embedding = OnehotEmbedding(20,5)
		phi_train = self.Embedding.embed(self.x_train)
		k = KernelFunction(kernel_name = "ard",kappa = 3., ard_gamma = torch.ones(100).double()*0.01, d = 100)
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s)
		self.GP.fit_gp(phi_train,self.y_train)
		self.GP.optimize_params(type="bandwidth", restarts=self.restarts, verbose = True, maxiter=self.maxiter, mingradnorm = 10e-5, optimizer = 'pytorch-minimize', scale = 10)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True

	def predict(self):
		self.mu, self.std = self.GP.mean_std(self.Embedding.embed(self.x_test))
		return self.mu, self.std


class FullCovarLearner(ARDModelLearner):

	def __str__(self):
		return "full_covar"

	def load(self):
		pass

	def fit(self):
		self.Embedding = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.Embedding.load_projection("data/projection-dim5-norm.pt")
		d = self.Embedding.projected_components
		phi_train = self.Embedding.embed(self.x_train)

		k = KernelFunction(kernel_name = "full_covariance_se",kappa = 3., cov = torch.eye(d).double()*0.01, d = d)
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s)
		self.GP.fit_gp(phi_train,self.y_train)
		init_func = lambda x: torch.eye(d).double()*1
		self.GP.optimize_params_general(
			params={'0': {"cov": (init_func, Euclidean(d**2), None)}},
			restarts=self.restarts, verbose=True, maxiter=self.maxiter, mingradnorm=10e-5, optimizer='pytorch-minimize',
			scale=100)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True

