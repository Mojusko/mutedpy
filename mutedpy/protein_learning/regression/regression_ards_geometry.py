import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import pickle
import os
import numpy as np
import pandas as pd

# loaders
from mutedpy.utils.loaders.loader_basel import BaselLoader

# utils
from stpy.test_functions.protein_benchmark import ProteinOperator
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations

# scores

# processing

# models
from stpy.kernel import KernelFunction
from stpy.regression.gauss_procc import GaussianProcess
from stpy.embeddings.polynomial_embedding import CustomEmbedding

from pymanopt.manifolds import Euclidean

from mutedpy.protein_learning.regression.regression_ards import ARDModelLearner

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor


class ARDGeometricFeatures(ARDModelLearner):

	def __str__(self):
		return "ard_geometric"

	def load(self):

		if not self.loaded:
			self.EmbeddingG = ContactMap(self.data_folder+"new_features_volume.csv", truncation=False)
			self.EmbeddingG.normalize(xtest=True)
			self.EmbeddingG.restrict_to_varaint(std=0.0001)
			#self.EmbeddingG.pca(std=0.01)
			#self.EmbeddingG.normalize(xtest=True)

			# self.EmbeddingA = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
			# self.EmbeddingA.load_projection("data/projection-dim5-norm.pt")

			# self.Embedding = AdditiveEmbeddings([self.EmbeddingG,self.EmbeddingA],[self.EmbeddingG.m,self.EmbeddingA.projected_components])
			# self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])
			# self.Embedding = CustomEmbedding(5,self.embed,self.EmbeddingG.m+self.EmbeddingA.projected_components)
			self.Embedding = self.EmbeddingG
			self.loaded = True

	def effective_dim(self, x):
		n = x.size()[0]
		phi = self.Embedding.embed(x)
		_, K = self.GP.execute(phi)
		dim = torch.trace(K @ torch.inverse(K + torch.eye(n).double()))
		return dim

	def just_fit(self):
		phi_train = self.Embedding.embed(self.x_train)
		self.GP.fit_gp(phi_train, self.y_train)
	def embed(self, x):
		phi_test = self.Embedding.embed(x)
		return phi_test

	def fit(self, optimize=True):
		self.load()
		phi_train = self.Embedding.embed(self.x_train)
		d = self.Embedding.m
		k = KernelFunction(kernel_name="ard", kappa=3.,
						   ard_gamma=torch.ones(d).double() * 0.01,
						   d=d)

		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s=self.s, loss = self.loss)
		self.GP.fit_gp(phi_train, self.y_train)
		if optimize:
			self.GP.optimize_params(type="bandwidth", restarts=self.restarts, verbose=2, maxiter=self.maxiter,
									mingradnorm=10e-5, optimizer='pytorch-minimize', scale=1, save=True,
									init_func=self.init_func,save_name= self.results_folder + str(self) + "/model_" + str(self.split) + ".np")

		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True

	def feature_importance(self):
		return self.GP.kernel_object.params_dict['0']['ard_gamma']




class FullCovarLearnerGeometric(ARDGeometricFeatures):

	def __str__(self):
		return "full_covar_geometric"

	def load(self):
		self.Embedding = ContactMap(self.features, truncation=False)
		if self.pca:
			self.Embedding.pca(std=self.err)
		else:
			self.Embedding.restrict_to_varaint(std=self.err)

		self.Embedding.normalize(xtest=True)

	def fit(self):
		self.load()
		d = self.Embedding.m
		phi_train = self.Embedding.embed(self.x_train)
		k = KernelFunction(kernel_name="full_covariance_se", kappa=3., cov=torch.eye(d).double() * 0.01, d=d)
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s=self.s)
		self.GP.fit_gp(phi_train, self.y_train)
		k = d
		init_func = lambda x: torch.eye(d, k).double() * 0.01
		self.GP.optimize_params_general(
			params={'0': {"cov": (init_func, Euclidean(d * k), None)}},
			restarts=self.restarts, verbose=True, maxiter=self.maxiter, mingradnorm=10e-5, optimizer='pytorch-minimize',
			scale=100)
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True


class LassoFeatureSelectorARDGeometric(ARDGeometricFeatures):

	def __str__(self):
		prefix = self.loss
		if self.features_V:
			prefix += 'volume_'
		if self.features_G:
			prefix += 'geometric_'
		if self.features_A:
			prefix += "AA_"
		if self.pca:
			prefix += "pca_"
		return prefix + "ard_lasso_features_" + str(self.topk) + "_kernel_" + self.ard_kernel



	# def preload(self):
	# 	self.list_of_embeddings = []
	# 	self.list_of_embedding_names = []
	#
	# 	if self.features_V:
	# 		self.EmbeddingV = ContactMap(self.data_folder + "new_features_volume.csv", truncation=False)
	# 		if self.pca:
	# 			self.EmbeddingV.pca(std=0.01, relative_var=True)
	# 			self.EmbeddingV.normalize(xtest=True)
	# 		else:
	# 			self.EmbeddingV.normalize(xtest=True)
	# 			self.EmbeddingV.restrict_to_varaint(std=0.001)
	#
	# 		if self.feature_split:
	# 			self.EmbeddingV.split_features(xtest=True)
	#
	# 		self.list_of_embeddings.append(self.EmbeddingV)
	# 		self.list_of_embedding_names.append(self.EmbeddingV.feature_names)
	#
	# 	if self.features_A:
	# 		self.EmbeddingA = AminoAcidEmbedding(data=self.data_folder + "amino-acid-features.csv")
	# 		if self.pca:
	# 			self.EmbeddingA.load_projection(self.data_folder + "projection-dim5-demean-norm.pt")
	# 		else:
	# 			self.EmbeddingA.load_projection(self.data_folder + "embedding-dim5-demean-norm.pt")
	# 			self.EmbeddingA.set_proto_names(5)
	#
	# 		if self.feature_split:
	# 			self.EmbeddingA.split_features(xtest=True, n_sites = 5)
	#
	# 		self.list_of_embeddings.append(self.EmbeddingA)
	# 		self.list_of_embedding_names.append(self.EmbeddingA.feature_names)
	#
	# 	if self.features_G:
	# 		self.EmbeddingG = ContactMap(self.data_folder + "new_features_rosetta_small.csv", truncation=False)
	# 		if self.pca:
	# 			self.EmbeddingG.pca(std=0.01, relative_var=True)
	# 			self.EmbeddingG.normalize(xtest=True)
	# 		else:
	# 			self.EmbeddingG.normalize(xtest=True)
	# 			self.EmbeddingG.restrict_to_varaint(std=0.001)
	#
	# 		if self.feature_split:
	# 			self.EmbeddingG.split_features(xtest=True)
	#
	#
	# 		self.list_of_embeddings.append(self.EmbeddingG)
	# 		self.list_of_embedding_names.append(self.EmbeddingG.feature_names)
	#
	# 	self.Embedding = AdditiveEmbeddings([self.list_of_embeddings],
	# 										[e.m for e in self.list_of_embeddings])
	#
	# 	self.embed = lambda x: torch.hstack(
	# 		[e.embed(x) for e in self.list_of_embeddings])
	# 	self.list_of_embedding_names = np.concatenate(self.list_of_embedding_names)

	def preload(self):
		self.Embedding = self.feature_loader

	def load(self):
		self.preload()
		if not self.loaded:

			# all phis to do model selection?
			phi_all = self.embed(self.x)

			self.feature_selector.pass_data(phi_all, self.y)
			self.feature_mask = self.feature_selector.select(self.topk)

			self.Embedding_original = self.Embedding
			embed = lambda x: self.Embedding_original.embed(x)[:,self.feature_mask]
			d = torch.sum(self.feature_mask)

			self.Embedding = CustomEmbedding(5,embed,d)
			print ("Final dim:", d)
			self.feature_names = self.Embedding_original.feature_names
			#print ( self.feature_names)
			self.loaded = True

	def fit(self, optimize=True, save_loc = None):
		self.load()
		phi_train = self.Embedding.embed(self.x_train)
		d = self.Embedding.m
		k = KernelFunction(kernel_name=self.ard_kernel, kappa=3.,
						   ard_gamma=torch.ones(d).double() * 0.01,
						   d=d, nu = 2.5)

		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s=self.s, loss = self.loss)
		self.GP.fit_gp(phi_train, self.y_train)

		if save_loc is None:
			save_loc = self.results_folder + str(self) + "/model_" + str(self.split) + ".np"

		if optimize:
			if self.ard_kernel != "full_covariance_se":
				self.GP.optimize_params(type="bandwidth", restarts=self.restarts, verbose=2, maxiter=self.maxiter,
									mingradnorm=10e-5, optimizer='pytorch-minimize', scale=10., save=True,
									init_func=self.init_func,save_name= save_loc)
			else:
				init_func = lambda x: torch.eye(d).double() * 1.
				self.GP.optimize_params_general(
					params={'0': {"cov": (init_func, Euclidean(d * d), None)}},
					restarts=self.restarts, verbose=True, maxiter=self.maxiter, mingradnorm=10e-5,
					optimizer='pytorch-minimize',save_name=save_loc)
				
				
		self.GP.fit_gp(phi_train, self.y_train)
		self.fitted = True

	def save_model(self, id = "", save_loc = None, special_identifier = ''):
		d = {}
		# selected features
		d['feature_mask'] = self.feature_mask
		d['feature_names'] = self.feature_names
		# what kernel
		d['kernel'] = self.ard_kernel
		# gamma
		d['kernel_object'] = self.GP.kernel_object
		d['ard_gamma'] = self.GP.kernel_object.ard_gamma
		# noise
		d['noise_std'] = self.GP.s
		d['x'] = self.GP.x
		d['y'] = self.GP.y
		if save_loc is None:
			filename = self.results_folder + str(self) + "/model_params_" + id + special_identifier + ".p"
		else:
			filename = save_loc

		with open(filename, "wb") as f:
			pickle.dump(d, f)


class RFFeatureSelectorARDGeometric(LassoFeatureSelectorARDGeometric):

	def __str__(self):
		prefix = self.loss
		if self.features_V:
			prefix += 'volume'
		if self.features_G:
			prefix += 'geometric'
		if self.features_A:
			prefix += "AA"
		if self.pca:
			prefix += "pca_"
		return prefix+"ard_rf_features_" + str(self.topk) + "_kernel_" + self.ard_kernel

	def load(self):
		if not self.loaded:
			self.preload()

			self.estimate_noise_std()
			phi_all = self.embed(self.x)

			self.regr = RandomForestRegressor(max_features='auto', max_depth=15, random_state=0, min_samples_split=5,
											  n_estimators=15000,  n_jobs=self.njobs)

			self.regr.fit(phi_all.numpy(), self.y.numpy().ravel())
			coef_ = self.regr.feature_importances_ / np.sum(self.regr.feature_importances_)

			d = self.topk
			self.feature_mask = torch.topk(torch.abs(torch.from_numpy(coef_)),k=self.topk)[1]
			self.embed = lambda x: torch.hstack([e.embed(x) for e in self.list_of_embeddings])[:,self.feature_mask]
			self.Embedding = CustomEmbedding(5,self.embed,d)

			print ("Final dim:", d)
			self.feature_names = self.list_of_embedding_names[self.feature_mask]
			print ( self.feature_names)
			self.loaded = True


class LassoRFFeatureSelectorARDGeometric(LassoFeatureSelectorARDGeometric):

	def __str__(self):
		prefix = self.loss
		if self.features_V:
			prefix += 'volume'
		if self.features_G:
			prefix += 'geometric'
		if self.features_A:
			prefix += "AA"
		if self.pca:
			prefix += "pca_"


		return prefix+"ard_rf+lasso_features_" + str(self.topk) + "_kernel_" + self.ard_kernel

	def load(self):
		if not self.loaded:
			self.preload()

			self.estimate_noise_std()
			phi_all = self.embed(self.x)

			self.regr = RandomForestRegressor(max_features='auto', max_depth=15, random_state=0, min_samples_split=5,
											  n_estimators=15000,  n_jobs=28)

			self.regr.fit(phi_all.numpy(), self.y.numpy().ravel())
			coef_ = self.regr.feature_importances_ / np.sum(self.regr.feature_importances_)
			n =  self.regr.feature_importances_.shape[0]
			self.feature_mask = torch.topk(torch.abs(torch.from_numpy(coef_)),k=self.topk)[1]

			self.regr = LassoCV(cv=10, n_alphas=200, random_state=0, max_iter=5000).fit(phi_all.numpy(),
																					   self.y.numpy().ravel())
			self.feature_mask2 = torch.topk(torch.abs(torch.from_numpy(self.regr.coef_)), k=self.topk)[1]

			self.feature_mask = torch.hstack((self.feature_mask,self.feature_mask2))
			self.feature_mask = torch.unique(self.feature_mask)
			print (self.feature_mask)

			d = self.feature_mask.size()[0]
			self.embed = lambda x: torch.hstack([e.embed(x) for e in self.list_of_embeddings])[:,self.feature_mask]
			self.Embedding = CustomEmbedding(5,self.embed,d)
			self.feature_names = self.list_of_embedding_names[self.feature_mask]
			print ( self.feature_names)
			self.loaded = True







if __name__ == "__main__":

	# Load data

	filename = "../../../data/streptavidin/5sites.xls"
	loader = BaselLoader(filename)
	dts = loader.load()

	filename = "../../../data/streptavidin/2sites.xls"
	loader = BaselLoader(filename)
	total_dts = loader.load(parent='SK', positions=[112, 121])
	total_dts = loader.add_mutations('T111T+N118N+A119A', total_dts)

	total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)
	total_dts = drop_neural_mutations(total_dts)
	total_dts['LogFitness'] = np.log10(total_dts['Fitness'])

	Op = ProteinOperator()
	x = torch.from_numpy(Op.translate_mutation_series(total_dts['variant']))
	y = torch.from_numpy(total_dts['LogFitness'].values).view(-1, 1)

	# initialize the model
	models = []
	restarts = 2
	maxiter = 200
	splits = 5

	lasso_params = [
		{'kernel':'ard', 'topk':10},
		{'kernel':'ard', 'topk':15},
		{'kernel':'ard', 'topk':20},
		{'kernel':'ard', 'topk':50},
		{'kernel':'ard_matern', 'topk': 10},
		{'kernel':'ard_matern', 'topk': 15},
		{'kernel':'ard_matern', 'topk': 20},
		{'kernel':'ard_matern', 'topk':50},
		{'kernel': 'ard', 'topk': 10, 'features_V': False, 'features_G': False},
		{'kernel': 'ard', 'topk': 15, 'features_V': False, 'features_G': False},
		{'kernel': 'ard', 'topk': 20, 'features_V': False, 'features_G': False},
		{'kernel': 'ard', 'topk': 50, 'features_V': False, 'features_G': False},
		{'kernel': 'ard_matern', 'topk': 10, 'features_V':False, 'features_G':False},
		{'kernel': 'ard_matern', 'topk': 15,'features_V':False, 'features_G':False},
		{'kernel': 'ard_matern', 'topk': 20,'features_V':False, 'features_G':False},
		{'kernel': 'ard_matern', 'topk': 50,'features_V':False, 'features_G':False},
		{'kernel': 'ard', 'topk': 10, 'pca':True},
		{'kernel': 'ard', 'topk': 15, 'pca':True},
		{'kernel': 'ard', 'topk': 20, 'pca':True},
		{'kernel': 'ard', 'topk': 50, 'pca':True},
		{'kernel': 'ard_matern', 'topk': 10, 'pca':True},
		{'kernel': 'ard_matern', 'topk': 15, 'pca':True},
		{'kernel': 'ard_matern', 'topk': 20, 'pca':True},
		{'kernel': 'ard_matern', 'topk': 50, 'pca':True},
		{'kernel': 'ard', 'topk': 10, 'features_V': False, 'features_G': False, 'pca':True},
		{'kernel': 'ard', 'topk': 15, 'features_V': False, 'features_G': False, 'pca':True},
		{'kernel': 'ard', 'topk': 20, 'features_V': False, 'features_G': False, 'pca':True},
		{'kernel': 'ard', 'topk': 50, 'features_V': False, 'features_G': False, 'pca':True},
		{'kernel': 'ard_matern', 'topk': 10, 'features_V': False, 'features_G': False, 'pca':True},
		{'kernel': 'ard_matern', 'topk': 15, 'features_V': False, 'features_G': False, 'pca':True},
		{'kernel': 'ard_matern', 'topk': 20, 'features_V': False, 'features_G': False, 'pca':True},
		{'kernel': 'ard_matern', 'topk': 50, 'features_V': False, 'features_G': False, 'pca':True}
	]
	params = lasso_params + lasso_params + lasso_params

	lasso_models = [LassoFeatureSelectorARDGeometric for _ in range(len(lasso_params))]
	rf_models = [RFFeatureSelectorARDGeometric for _ in range(len(lasso_params))]
	lasso_rf_models = [LassoRFFeatureSelectorARDGeometric for _ in range(len(lasso_params)) ]

	models = lasso_models + rf_models + lasso_rf_models
	default_params = {'restarts':restarts, 'data_folder':"../data/",'results_folder':"../results_strep/","maxiter":maxiter}

	for param, model in zip(params, models):
		#try:
		model = model(** {**default_params, **param})
		model.add_data(x, y)
		# create folder for results_strep
		try:
			os.mkdir(default_params['results_folder']+ str(model))
		except:
			print ("Folder already exists.")

		# try loading it
		model.evaluate_metrics_on_splits(no_splits=splits,
										 split_location = "../splits/random_splits.pt")
