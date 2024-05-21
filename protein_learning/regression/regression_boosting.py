import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
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
from metricpy.aminoacid_embedding import AminoAcidEmbedding, ContactMap
from stpy.embeddings.embedding import AdditiveEmbeddings
from sklearn.ensemble import RandomForestRegressor
from mutedpy.protein_learning.gaussian_process.regression_basis import ProteinKernelLearner
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

class XGboostGeometric(ProteinKernelLearner):

	def __str__(self):
		return "xgboost-geometric-aa"

	def fit(self):


		self.EmbeddingG = ContactMap("../data/new_features_rosetta.csv", truncation=False)
		self.EmbeddingG.normalize(xtest=True)
		self.EmbeddingG.restrict_to_varaint(std=0.00001)
		self.EmbeddingG.pca(std = 0.01)

		self.EmbeddingA = AminoAcidEmbedding(data = "../data/amino-acid-features.csv")
		self.EmbeddingA.load_projection("../data/projection-dim5-norm.pt")

		# self.EmbeddingA = ContactMap("data/features.csv", truncation=False)
		# self.EmbeddingA.normalize(xtest=True)
		# self.EmbeddingA.restrict_to_varaint(std=0.0005)

		self.Embedding = AdditiveEmbeddings([self.EmbeddingG,self.EmbeddingA],[self.EmbeddingG.m,self.EmbeddingA.projected_components])
		self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])

		phi_train = self.embed(self.x_train)
		d = self.Embedding.m

		self.estimate_noise_std()
		#self.regr = GradientBoostingRegressor(max_features="auto", random_state=0, n_estimators = 1000, max_depth = 3)
		self.regr = xgb.XGBRegressor(max_depth=5, n_estimators = 10000, n_jobs = 28)

		self.regr.fit(phi_train.numpy(),self.y_train.numpy().ravel())
		self.fitted = True

		# print feature importance
		dt = pd.DataFrame([self.regr.feature_importances_, np.concatenate((self.EmbeddingG.feature_names,self.EmbeddingA.feature_names))]).T
		print (dt)
		dt.to_csv('../results_strep/fea_imp-xgboost-geometric-aa.csv')

	def predict(self):
		mu = self.regr.predict(self.embed(self.x_test).numpy())
		std = mu*0
		self.mu = torch.from_numpy(mu)
		self.std = torch.from_numpy(std)

		return self.mu, self.std


class RandomForestLearner(ProteinKernelLearner):

	def __str__(self):
		return "random-forest"

	def fit(self):
		self.Embedding = AminoAcidEmbedding(data="data/amino-acid-features.csv")
		self.Embedding.load_projection("data/projection-dim5-norm.pt")
		d = self.Embedding.projected_components
		phi_train = self.Embedding.embed(self.x_train)

		self.estimate_noise_std()
		self.regr = RandomForestRegressor(max_depth=8, random_state=0, min_samples_split=10, n_estimators = 1000)
		self.regr.fit(phi_train.numpy(),self.y_train.numpy())
		self.fitted = True

		# print feature importance
		np.savetxt('results_strep/fea_imp.txt',self.regr.feature_importances_)

	def predict(self):
		mu = self.regr.predict(self.Embedding.embed(self.x_test).numpy())
		std = mu*0 + 1
		self.mu = torch.from_numpy(mu)
		self.std = torch.from_numpy(std)

		return self.mu, self.std



class RandomForestLearnerGeometric(ProteinKernelLearner):

	def __str__(self):
		return "random-forest-geometric"

	def fit(self):


		self.Embedding = ContactMap("data/features.csv", truncation=False)
		self.Embedding.normalize(xtest=True)
		self.Embedding.restrict_to_varaint(std=0.0005)

		phi_train = self.Embedding.embed(self.x_train)
		d = self.Embedding.m

		self.estimate_noise_std()
		self.regr = RandomForestRegressor(max_depth=15, random_state=0, min_samples_split=5, n_estimators = 1000)
		self.regr.fit(phi_train.numpy(),self.y_train.numpy())
		self.fitted = True

		# print feature importance
		dt = pd.DataFrame([self.regr.feature_importances_, self.Embedding.feature_names]).T
		print (dt)
		dt.to_csv('results_strep/fea_imp.csv')

	def predict(self):
		mu = self.regr.predict(self.Embedding.embed(self.x_test).numpy())
		std = mu*0 + 1
		self.mu = torch.from_numpy(mu)
		self.std = torch.from_numpy(std)

		return self.mu, self.std


class RandomForestLearnerGeometricAA(ProteinKernelLearner):

	def __str__(self):
		prefix = self.loss+"_"
		if self.features_V:
			prefix += 'volume_'
		if self.features_G:
			prefix += 'geometric_'
		if self.features_A:
			prefix += "AA_"
		if self.pca:
			prefix += "pca_"
		return prefix+"random-forests"

	def fit(self):
		self.list_of_embeddings = []
		if self.features_G:
			self.EmbeddingG = ContactMap("../data/new_features_rosetta.csv", truncation=False)
			if self.pca:
				self.EmbeddingG.pca(std = 0.01, relative_var = True, name= "geo")
				self.EmbeddingG.normalize(xtest=True)
			else:
				self.EmbeddingG.normalize(xtest=True)
				self.EmbeddingG.restrict_to_varaint(std=0.0000001)
			self.list_of_embeddings.append(self.EmbeddingG)
		if self.features_V:
			self.EmbeddingV = ContactMap("../data/new_features_volume.csv", truncation=False)
			if self.pca:
				self.EmbeddingV.pca(std = 0.01, relative_var = True, name = "volume")
				self.EmbeddingV.normalize(xtest=True)
			else:
				self.EmbeddingV.normalize(xtest=True)
				self.EmbeddingV.restrict_to_varaint(std=0.0000001)
			self.list_of_embeddings.append(self.EmbeddingV)
		if self.features_A:
			self.EmbeddingA = AminoAcidEmbedding(data = "../data/amino-acid-features.csv")
			if self.pca:
				self.EmbeddingA.load_projection("../data/projection-dim5-demean-norm.pt")
			else:
				self.EmbeddingA.load_projection("../data/embedding-dim5-demean-norm.pt")
			self.list_of_embeddings.append(self.EmbeddingA)

		self.Embedding = AdditiveEmbeddings([e for e in self.list_of_embeddings],
											[e.m for e in self.list_of_embeddings])

		self.embed = lambda x: torch.hstack([e.embed(x) for e in self.list_of_embeddings])

		phi_train = self.embed(self.x_train)

		self.estimate_noise_std()
		self.regr = RandomForestRegressor(max_features="auto",max_depth=15,
										  random_state=0, min_samples_split=5,
										  n_estimators = 10000, n_jobs=28, criterion=self.loss, verbose=True)
		self.regr.fit(phi_train.numpy(),self.y_train.numpy().ravel())
		self.fitted = True

		# print feature importance
		dts = pd.DataFrame([self.regr.feature_importances_, np.concatenate([e.feature_names for e in self.list_of_embeddings])]).T
		print (dts)
		dts.to_csv(self.results_folder+'fea_imp-geometric'+str(self)+'-'+str(self.split)+'.csv')

	def predict(self):
		mu = self.regr.predict(self.embed(self.x_test).numpy())
		mu_train = self.regr.predict(self.embed(self.x_train).numpy())

		std = mu*0
		std_train = std *0

		self.mu = torch.from_numpy(mu)
		self.std = torch.from_numpy(std)
		self.mu_train = torch.from_numpy(mu_train)
		self.std_train = torch.from_numpy(std_train)
		return self.mu, self.std

class XGboost(ProteinKernelLearner):

	def __str__(self):
		return "xgboost-aa"

	def fit(self):


		# self.EmbeddingG = ContactMap("data/new_features.csv", truncation=False)
		# self.EmbeddingG.normalize(xtest=True)
		# self.EmbeddingG.restrict_to_varaint(std=0.000001)

		self.EmbeddingA = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.EmbeddingA.load_projection("data/projection-dim5-norm.pt")

		# self.EmbeddingA = ContactMap("data/features.csv", truncation=False)
		# self.EmbeddingA.normalize(xtest=True)
		# self.EmbeddingA.restrict_to_varaint(std=0.0005)

		self.Embedding = self.EmbeddingA
		self.embed = lambda x: self.EmbeddingA.embed(x)

		phi_train = self.embed(self.x_train)
		d = self.Embedding.m

		self.estimate_noise_std()
		#self.regr = GradientBoostingRegressor(max_features="auto", random_state=0, n_estimators = 10000, max_depth = 10)
		self.regr = xgb.XGBRegressor(max_depth = 10, tree_method = "hist",  n_estimators = 10000, n_jobs = 28)

		self.regr.fit(phi_train.numpy(),self.y_train.numpy().ravel())
		self.fitted = True

		# print feature importance
		dt = pd.DataFrame([self.regr.feature_importances_, self.EmbeddingA.feature_names]).T
		print (dt)
		dt.to_csv('results_strep/fea_imp-xgboost-aa.csv')

	def predict(self):
		mu = self.regr.predict(self.embed(self.x_test).numpy())
		std = mu*0
		self.mu = torch.from_numpy(mu)
		self.std = torch.from_numpy(std)

		return self.mu, self.std


class NeuralNet(ProteinKernelLearner):

	def __str__(self):
		return "xgboost-geometric-aa"

	def fit(self):


		self.EmbeddingG = ContactMap("data/new_features.csv", truncation=False)
		self.EmbeddingG.normalize(xtest=True)
		self.EmbeddingG.restrict_to_varaint(std=0.00001)

		self.EmbeddingA = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		#self.EmbeddingA.load_projection("data/projection-dim5-norm.pt")

		# self.EmbeddingA = ContactMap("data/features.csv", truncation=False)
		# self.EmbeddingA.normalize(xtest=True)
		# self.EmbeddingA.restrict_to_varaint(std=0.0005)

		self.Embedding = AdditiveEmbeddings([self.EmbeddingG,self.EmbeddingA],[self.EmbeddingG.m,self.EmbeddingA.projected_components])
		self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])

		phi_train = self.embed(self.x_train)
		d = self.Embedding.m

		self.estimate_noise_std()
		self.regr = MLPRegressor(learning_rate_init = 0.001, hidden_layer_sizes = [512,128,64,32,16], verbose=True, early_stopping = True)
		self.regr.fit(phi_train.numpy(),self.y_train.numpy().ravel())
		self.fitted = True

		# print feature importance
		# dt = pd.DataFrame([self.regr.feature_importances_, np.concatenate((self.EmbeddingG.feature_names,self.EmbeddingA.feature_names))]).T
		# print (dt)
		# dt.to_csv('results_strep/fea_imp-xgboost-geometric-aa.csv')

	def predict(self):
		mu = self.regr.predict(self.embed(self.x_test).numpy())
		std = mu*0
		self.mu = torch.from_numpy(mu)
		self.std = torch.from_numpy(std)

		return self.mu, self.std



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
	restarts = 1
	maxiter = 20
	splits = 5

	models = [RandomForestLearnerGeometricAA]

	default_params = {'restarts':restarts, 'data_folder':"../data/",'results_folder':"../results_strep/","maxiter":maxiter,
					  'loss':"mae"}

	for model in models:
		model = model(**default_params)
		model.add_data(x, y)
		# create folder for results_strep
		try:
			os.mkdir(default_params['results_folder'] + str(model))
		except:
			print("Folder already exists.")

		# try loading it
		model.evaluate_metrics_on_splits(no_splits=splits, split_location="../splits/random_splits.pt")