import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import numpy as np
import pandas as pd
import xgboost as xgb

# loaders
from mutedpy.utils.loaders.loader_basel import BaselLoader

# utils
from stpy.test_functions.protein_benchmark import ProteinOperator
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations

# scores

# processing

# models
from metricpy.aminoacid_embedding import AminoAcidEmbedding, ContactMap
from stpy.kernels import KernelFunction
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.embeddings.polynomial_embedding import CustomEmbedding
from stpy.embeddings.embedding import AdditiveEmbeddings
from sklearn.ensemble import RandomForestRegressor
from mutedpy.protein_learning.gaussian_process.regression_basis import ProteinKernelLearner,ARDGeometricFeatures, FullCovarLearnerGeometric, PairwiseAllARDModelLearning, AdditiveARDModelLearning
from sklearn.linear_model import LassoCV, SGDRegressor


class HuberLinearModelLearner(LassoModelLearner):

	def __str__(self):
		return "huber_linear_geometric_rosetta_volume"

	def fit(self):
		self.EmbeddingV = ContactMap("../data/new_features_volume.csv", truncation=False)
		self.EmbeddingV.normalize(xtest=True)
		self.EmbeddingV.restrict_to_varaint(std=0.0000001)

		self.EmbeddingA = AminoAcidEmbedding(data = "../data/amino-acid-features.csv")
		self.EmbeddingA.load_projection("../data/projection-dim5-norm.pt")

		self.EmbeddingG = ContactMap(data = "../data/new_features_rosetta.csv", truncation=False)
		self.EmbeddingG.normalize(xtest=True)
		self.EmbeddingG.restrict_to_varaint(std=0.0000001)

		self.Embedding = AdditiveEmbeddings([self.EmbeddingG,self.EmbeddingV,self.EmbeddingA],[self.EmbeddingG.m,self.EmbeddingV.m,self.EmbeddingA.projected_components])
		self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x),self.EmbeddingV.embed(x), self.EmbeddingA.embed(x)])

		self.estimate_noise_std()
		phi_train = self.embed(self.x_train)
		self.regr = SGDRegressor(loss = 'huber', penalty = 'l1', max_iter = 100000, alpha = 0.01).fit(phi_train.numpy(), self.y_train.numpy())

		dts = pd.DataFrame([self.regr.coef_, np.concatenate((self.EmbeddingG.feature_names,
															 self.EmbeddingV.feature_names,
															 self.EmbeddingA.feature_names))]).T

		#dts = pd.DataFrame([self.regr.coef_, self.EmbeddingG.feature_names]).T

		dts.to_csv("../results_strep/lasso-features"+str(self.split)+".csv")
		self.fitted = True

class LassoAndARDCorrection(ProteinKernelLearner):

	def __str__(self):
		return "lasso_geometric_ard_normal_correction"

	def fit(self):
		self.EmbeddingG = ContactMap("data/new_features.csv", truncation=False)
		self.EmbeddingG.normalize(xtest=True)
		self.EmbeddingG.restrict_to_varaint(std=0.00001)

		self.EmbeddingA = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.EmbeddingA.load_projection("data/projection-dim5-norm.pt")

		self.embed = lambda x: self.EmbeddingG.embed(x)
		self.embed2 = lambda x: self.EmbeddingA.embed(x)

		self.estimate_noise_std()

		phi_train = self.embed(self.x_train)
		self.regr = LassoCV(cv=10,n_alphas=1000, random_state=0, max_iter = 5000).fit(phi_train.numpy(), self.y_train.numpy())

		dts = pd.DataFrame([self.regr.coef_, self.EmbeddingG.feature_names]).T
		dts.to_csv("results_strep/lasso-features.csv")

		self.y_train_new = self.y_train - torch.from_numpy(self.regr.predict(phi_train.numpy())).view(-1, 1)
		new_phi_train = self.embed2(self.x_train)

		d = self.EmbeddingA.projected_components
		k = KernelFunction(kernel_name="ard", kappa=3.,
						   ard_gamma=torch.ones(d).double() * 0.01,
						   d=d)

		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s=self.s)
		self.GP.fit_gp(new_phi_train, self.y_train_new)
		self.GP.optimize_params(type="bandwidth", restarts=self.restarts, verbose=True, maxiter=self.maxiter,
									mingradnorm=10e-5, optimizer='pytorch-minimize', scale=1000, save=True,
									init_func=self.init_func,
									save_name="model_" +str(self)+"_" + str(self.split) + ".np")
		self.GP.fit_gp(new_phi_train, self.y_train_new)
		self.fitted = True

	def predict(self):
		self.mu =  torch.from_numpy(self.regr.predict(self.embed(self.x_test).numpy())).view(-1,1) + self.GP.mean_std(self.embed2(self.x_test))[0]
		self.std = self.mu*0
		return self.mu, self.std




class SelectedFeaturePairwiseAllARDModelLearning(PairwiseAllARDModelLearning):

	def __str__(self):
		return "pairwise_ard_geometric_lasso_features"

	def load(self):
		self.EmbeddingG = ContactMap("data/new_features.csv", truncation=False)
		self.EmbeddingG.normalize(xtest=True)
		self.EmbeddingG.restrict_to_varaint(std=0.00001)

		#self.EmbeddingA = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		#self.EmbeddingA.load_projection("data/projection-dim5-norm.pt")

		# self.EmbeddingA = ContactMap("data/features.csv", truncation=False)
		# self.EmbeddingA.normalize(xtest=True)
		# self.EmbeddingA.restrict_to_varaint(std=0.0005)

		#self.Embedding = AdditiveEmbeddings([self.EmbeddingG,self.EmbeddingA],[self.EmbeddingG.m,self.EmbeddingA.projected_components])
		#self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])
		#d = self.Embedding.m

		self.estimate_noise_std()
		phi_all = self.EmbeddingG.embed(self.x)
		self.regr = LassoCV(cv=5, max_iter=2000,random_state=0).fit(phi_all.numpy(), self.y.numpy())

		mask = torch.abs(torch.from_numpy(self.regr.coef_)) > 0.1
		self.embed = lambda x: self.EmbeddingG.embed(x)[:,mask]
		d = torch.sum(mask)
		self.Embedding = CustomEmbedding(5,self.embed,d)
		print ("New dim:", d)



class SelectedFeaturesAdditiveARDModelLearning(AdditiveARDModelLearning):

	def __str__(self):
		return "additive_ard_geometric_lasso_features"

	def load(self):
		self.EmbeddingG = ContactMap("data/new_features.csv", truncation=False)
		self.EmbeddingG.normalize(xtest=True)
		self.EmbeddingG.restrict_to_varaint(std=0.00001)

		#self.Embedding = AdditiveEmbeddings([self.EmbeddingG,self.EmbeddingA],[self.EmbeddingG.m,self.EmbeddingA.projected_components])
		#self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])
		#d = self.Embedding.m

		self.estimate_noise_std()
		phi_all = self.EmbeddingG.embed(self.x)
		self.regr = LassoCV(cv=5, max_iter=5000,random_state=0).fit(phi_all.numpy(), self.y.numpy())
		coef = self.regr.coef_
		k = 20
		mask = torch.topk(torch.abs(torch.from_numpy(coef)), k=k)[1]

		self.embed = lambda x: self.EmbeddingG.embed(x)[:,mask]
		d = k
		self.Embedding = CustomEmbedding(5,self.embed,d)
		print ("New dim:", d)
		self.groups = [[i] for i in range(d)]

class RFFeatureSelectorARDGeometric(ARDGeometricFeatures):

	def __str__(self):
		return "ard_geometric_rf_features"

	def load(self):
		self.EmbeddingG = ContactMap("data/new_features.csv", truncation=False)
		self.EmbeddingG.normalize(xtest=True)
		self.EmbeddingG.restrict_to_varaint(std=0.00001)

		self.EmbeddingA = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.EmbeddingA.load_projection("data/projection-dim5-norm.pt")

		# self.EmbeddingA = ContactMap("data/features.csv", truncation=False)
		# self.EmbeddingA.normalize(xtest=True)
		# self.EmbeddingA.restrict_to_varaint(std=0.0005)

		self.Embedding = AdditiveEmbeddings([self.EmbeddingG,self.EmbeddingA],[self.EmbeddingG.m,self.EmbeddingA.projected_components])
		self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])
		d = self.Embedding.m

		self.estimate_noise_std()
		phi_all = self.embed(self.x)
		self.regr = RandomForestRegressor(max_features='sqrt',max_depth=10, random_state=0, min_samples_split=5, n_estimators=50000,
										  n_jobs=28)
		self.regr.fit(phi_all.numpy(), self.y.numpy().ravel())
		coef = self.regr.feature_importances_/np.sum(self.regr.feature_importances_)
		k = 5
		mask = torch.topk(torch.abs(torch.from_numpy(coef)), k=k)[1]
		self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])[:,mask]
		d = k
		self.Embedding = CustomEmbedding(5,self.embed,d)
		print ("New dim:", d)




class XgboostFeatureSelectorARDGeometric(FullCovarLearnerGeometric):

	def __str__(self):
		return "ard_geometric_rf_features"

	def load(self):
		if not self.loaded:
			self.EmbeddingG = ContactMap("data/new_features.csv", truncation=False)
			self.EmbeddingG.normalize(xtest=True)
			self.EmbeddingG.restrict_to_varaint(std=0.00001)

			self.EmbeddingA = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
			self.EmbeddingA.load_projection("data/projection-dim5-norm.pt")

			self.Embedding = AdditiveEmbeddings([self.EmbeddingG,self.EmbeddingA],[self.EmbeddingG.m,self.EmbeddingA.projected_components])
			self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])
			d = self.Embedding.m

			self.estimate_noise_std()
			phi_all = self.embed(self.x)
			self.regr = xgb.XGBRegressor(max_depth = 10, n_estimators = 10000, n_jobs = 28)

			self.regr.fit(phi_all.numpy(), self.y.numpy().ravel())
			coef = self.regr.feature_importances_/np.sum(self.regr.feature_importances_)
			k = 15
			mask = torch.topk(torch.abs(torch.from_numpy(coef)), k=k)[1]
			names = list(self.EmbeddingG.feature_names) + list(self.EmbeddingA.feature_names)
			print ("Selected features:")
			print ([names[j] for j in mask])
			self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])[:,mask]
			d = k
			self.Embedding = CustomEmbedding(5,self.embed,d)
			self.loaded = True
			print ("New dim:", d)


class LassoFeatureSelectorFullGeometric(FullCovarLearnerGeometric):

	def __str__(self):
		return "full_geometric_lasso_features"

	def load(self):
		self.EmbeddingG = ContactMap("data/features.csv", truncation=False)
		self.EmbeddingG.normalize(xtest=True)
		self.EmbeddingG.restrict_to_varaint(std=0.00001)

		self.EmbeddingA = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
		self.EmbeddingA.load_projection("data/projection-dim5-norm.pt")

		# self.EmbeddingA = ContactMap("data/features.csv", truncation=False)
		# self.EmbeddingA.normalize(xtest=True)
		# self.EmbeddingA.restrict_to_varaint(std=0.0005)

		self.Embedding = AdditiveEmbeddings([self.EmbeddingG,self.EmbeddingA],[self.EmbeddingG.m,self.EmbeddingA.projected_components])
		self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])
		d = self.Embedding.m

		self.estimate_noise_std()
		phi_all = self.embed(self.x)
		self.regr = LassoCV(cv=5, random_state=0).fit(phi_all.numpy(), self.y.numpy())

		mask = torch.topk(torch.abs(torch.from_numpy(self.regr.coef_)), k = 20)[1]
		self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])[:,mask]
		d = torch.sum(mask)
		self.Embedding = CustomEmbedding(5,self.embed,d)
		print ("New dim:", d)



class RFFeatureSelectorFullGeometric(FullCovarLearnerGeometric):

	def __str__(self):
		return "full_geometric_rf_features"

	def load(self):

		if not self.loaded:
			self.EmbeddingG = ContactMap("data/features.csv", truncation=False)
			self.EmbeddingG.normalize(xtest=True)
			self.EmbeddingG.restrict_to_varaint(std=0.00001)

			self.EmbeddingA = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
			self.EmbeddingA.load_projection("data/projection-dim5-norm.pt")

			# self.EmbeddingA = ContactMap("data/features.csv", truncation=False)
			# self.EmbeddingA.normalize(xtest=True)
			# self.EmbeddingA.restrict_to_varaint(std=0.0005)

			self.Embedding = AdditiveEmbeddings([self.EmbeddingG,self.EmbeddingA],[self.EmbeddingG.m,self.EmbeddingA.projected_components])
			self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])
			d = self.Embedding.m

			self.estimate_noise_std()
			phi_all = self.embed(self.x)
			self.regr = RandomForestRegressor(max_depth=15, random_state=0, min_samples_split=5, n_estimators=5000,
											  n_jobs=28)
			self.regr.fit(phi_all.numpy(), self.y.numpy().ravel())
			coef = self.regr.feature_importances_/np.sum(self.regr.feature_importances_)
			k = 20
			mask = torch.topk(torch.abs(torch.from_numpy(coef)), k=k)[1]
			self.embed = lambda x: torch.hstack([self.EmbeddingG.embed(x), self.EmbeddingA.embed(x)])[:,mask]
			d = k
			self.Embedding = CustomEmbedding(5,self.embed,d)
			print ("New dim:", d)
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
	maxiter = 100
	splits = 5

	models = [HuberLinearModelLearner]

	for model in models:
		init_func = lambda m: torch.ones(size=(m, 1)).view(-1).double() * 1000 + torch.rand(size=(m, 1)).view(
			-1).double()
		model = model(restarts=restarts, maxiter=maxiter, err=0.001, init_func=None, pca=True)

		model.add_data(x, y)
		# mask, where = contact_map_embedding.mask_if_inside(model.x)
		# x = x[mask,:]
		# y = y[mask,:]
		# model.add_data(x,y)
		model.evaluate_metrics_on_splits(splits=splits, prefix="../plots", file="../results_strep/" + str(model) + ".csv", n_test=150)
		model.save_plot( prefix="../plots")