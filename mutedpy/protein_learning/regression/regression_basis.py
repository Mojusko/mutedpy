import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from scipy.stats import norm
import scipy
from abc import ABC
#import mkl
from multiprocessing.pool import Pool
# loaders

# utils 
from mutedpy.protein_learning.data_splits import load_splits
# scores
from mutedpy.utils.scoring.directed_evolution_scores import f1_score,enrichment_factor,enrichment_area,hit_rate

# processing
from mutedpy.utils.sequences.sequence_utils import from_variant_to_integer, from_integer_to_variants

# models
from mutedpy.protein_learning.featurizers.feature_loader import ProteinFeatureLoader, AddedProteinFeatureLoader

from stpy.kernel import KernelFunction
from stpy.regression.gauss_procc import GaussianProcess
from stpy.helpers.scores import r_score_std
from stpy.helpers.helper import estimate_std
from stpy.helpers.abitrary_sampling import randomly_split_set_without_duplicates
import sklearn
import copy

class ProteinKernelLearner(ABC):
	
	def __init__(self,
				 feature_loader = None,
				 feature_loader_caller_list=None,
				 feature_loader_params_list = [],
				 additive = False,
				 feature_selector = None,
				 feature_selector_caller=None,
				 feature_selector_params = {},
				 embed_type : Union['x','seq']= 'x',
				 model_selection_all = False,
				 restarts = 1,
				 maxiter = 10,
				 data_folder = "./",
				 features = "data/new_features.csv",
				 err = 0.1,
				 proj_dim = None,
				 init_func = None,
				 pca = False,
				 topk = 10,
				 kernel = "ard",
				 features_V = True,
				 features_G = True,
				 features_A = True,
				 results_folder = "./",
				 njobs = 4,
				 cores = 2,
				 feature_split = False,
				 loss = "squared",
				 prespecified_sigma = None,
				 threshold = 1.2
		) -> None:

		self.par = locals()

		if feature_selector is not None:
			self.feature_selector = feature_selector
		else:
			self.feature_selector = feature_selector_caller(**feature_selector_params)

		if feature_loader is not None:
			self.feature_loader = feature_loader
		else:
			self.feature_loader = AddedProteinFeatureLoader([
				fea(**param) for fea,param in zip(feature_loader_caller_list, feature_loader_params_list)])

		self.additive = additive
		self.threshold = threshold
		self.proj_dim = proj_dim
		self.embed_type = embed_type
		self.model_selection_all = model_selection_all
		self.restarts = restarts
		self.feature_mask = None
		self.njobs = njobs
		self.prespecified_sigma = prespecified_sigma
		self.feature_split = feature_split
		self.init_func = init_func
		self.pca = pca
		self.cores = cores
		self.loss = loss
		self.results_folder = results_folder
		self.ard_kernel = kernel
		self.topk = topk
		self.maxiter = maxiter
		self.data_folder = data_folder
		self.features = features
		self.err = err
		self.loaded = False
		self.split = -1

	def __str__(self):
		return "linear-kernel-amino-acid"

	def save_model(self, id = "", save_loc = None, special_identifier = ''):
		d = {}

		d['feature_mask'] = self.feature_mask

		# gamma
		d['kernel_object'] = self.GP.kernel_object

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
		return d

	def load(self):
		self.Embedding = self.feature_loader
		if self.seq is not None:
			self.x = self.embed(self.seq)

	def estimate_noise_std(self, truncation:Union[float,None] = None):

		if self.prespecified_sigma is not None:
			self.s = self.prespecified_sigma
			print ("Specified sigma:", self.s)
			return self.s
		else:
			sigma_std = estimate_std(self.x,self.y, truncation, verbose= True)
			self.s = sigma_std
			print ("Estimated sigma:", sigma_std)
			return sigma_std

	def add_data(self,x,y, seq = None):
		self.x = x 
		self.y = y

		if seq is not None:
			self.seq = seq
		else:
			self.seq = None


	def classic_eval(self, alpha = 0.95):
		rmsd = self.get_coverage()
		coverage = self.get_coverage(alpha = alpha)
		r2 = self.get_R2()
		r2std = self.get_R2std()
		pearson = self.get_pearson()
		spearman = self.get_spearman()
		hit = self.get_hit_rate()
		f1 = self.get_f1()
		ef = self.get_enrichment()
		ea = self.get_enrichment_area()
		return rmsd, coverage, r2, r2std, pearson, spearman, hit, f1, ef, ea

	def stratified_R2(self, val_min, val_max):
		mask = self.y_test>val_min
		mask2 = self.y_test<val_max
		mask = torch.logical_and(mask,mask2)
		if torch.sum(mask)<2:
			return torch.nan
		if self.fitted:
			r2 = sklearn.metrics.r2_score(self.y_test[mask], self.mu[mask])
			return r2
		else:
			return None

	def stratified_rmsd(self, val_min, val_max):
		mask = self.y_test>val_min
		mask2 = self.y_test<val_max
		mask = torch.logical_and(mask,mask2)
		if torch.sum(mask)<1:
			return torch.nan
		if self.fitted:
			RMSD = torch.mean((self.mu[mask] - self.y_test[mask])**2)
			return RMSD
		else:
			return None

	def stratified_pearson(self, val_min, val_max):
		mask = self.y_test>val_min
		mask2 = self.y_test<val_max
		mask = torch.logical_and(mask,mask2)
		if torch.sum(mask)<2:
			return torch.nan
		if self.fitted:
			r, p = scipy.stats.pearsonr(self.y_test[mask].detach().view(-1).numpy(), self.mu[mask].detach().view(-1).numpy())
			return r
		else:
			return None


	def production(self):
		self.x_train = self.x
		self.y_train = self.y
		self.seq_train = self.seq

	def split_data(self, n_test = 150):
		n = self.x.size()[0]
		mask_test, mask_train = randomly_split_set_without_duplicates(self.x,size = n_test)
		
		# test 
		self.x_test = self.x[mask_test,:]
		self.y_test = self.y[mask_test,:]

		# train 
		self.x_train = self.x[mask_train,:]
		self.y_train = self.y[mask_train,:]

		if self.seq is not None:
			self.seq_test = self.seq[mask_test]
			self.seq_train = self.seq[mask_train]

		return (self.x_test.clone(),self.y_test.clone(),self.x_train.clone(),self.y_train.clone())

	def cv_split_eval(self, splits = 10, n_test = 150):
		# create splits 
		dts = []
		for i in range(splits):
			d = self.split_data(n_test=n_test)
			dts.append(d)
		return dts

	def load_splits_eval(self, loc, splits = 10 ):
		dts = load_splits(splits, loc)
		return dts

	def complex_callibration(self):
		pass

	def evaluate_mutations_space_prediction_on_splits(self,
													  x_test,
													  y_test,
													  split_location = None,
													  no_splits = 10,
													  n_test = 150,
													  output_file = 'output.txt',
													  special_identifier = '',
													  F = None,
													  scoring = 'r2'):
		if split_location is None:
			dts = self.cv_split_eval(splits = no_splits, n_test = n_test)
		else:
			dts = self.load_splits_eval(split_location, splits=no_splits)


		splits = no_splits
		r2s = torch.zeros(size = (splits,1)).view(-1).double()
		alpha_range = np.arange(0.1,1.1,0.1)
		coverages = torch.zeros(size = (splits,10)).double()
		best_xs = torch.zeros(size = (splits,1)).view(-1).double()

		pearson = torch.zeros(size = (splits,1)).view(-1).double()

		def evaluate(v):
			coverage = torch.zeros(size=( 10,1)).double().view(-1)
			index, d = v
			print (index+1,"/", splits)
			self.x_test = x_test.clone()
			self.y_test = y_test.clone()

			# train
			self.x_train = d[2]
			self.y_train = d[3]
			self.split = index

			self.fit()
			self.predict()

			r2 = sklearn.metrics.r2_score(self.y_test, self.mu)
			pea = self.get_pearson()

			for j,alpha in enumerate(alpha_range):
				coverage[j] = self.get_coverage(alpha = alpha, predictive = False)

			F_max = torch.max(self.y_test)
			index_max = torch.argmax(self.mu)
			best_x = F_max - self.y_test[index_max]

			print ("r2:", r2)
			return r2, coverage, pea, best_x

		results = [evaluate(a) for a in enumerate(dts)]

		for index, v in enumerate(results):
			r2s[index] = v[0]
			coverages[index, :] = v[1]
			pearson[index] = v[2]
			best_xs[index] = v[3]

		names = ["r2","pearson","best_F"] + ["coverage_"+str(np.round(alpha,1)) for alpha in alpha_range]
		datas = [r2s, pearson,best_xs] + [coverages[:,j] for j in range(10)]

		self.save_results(names, datas, output_file = output_file)

	def get_splits(self):
		pass


	def evaluate_on_split(self, index, d, splits = 10,special_identifier="", save_model = True):
		coverage = torch.zeros(size=(10, 1)).double().view(-1)
		coverage_cons = torch.zeros(size=(10, 1)).double().view(-1)
		alpha_range = np.arange(0.1,1.1,0.1)

		torch.set_num_threads(self.njobs)
		#mkl.set_num_threads(self.njobs)

		print(index + 1, "/", splits)
		self.x_test = d[0]
		self.y_test = d[1]
		self.seq_test = from_integer_to_variants(d[0])

		# train
		self.x_train = d[2]
		self.y_train = d[3]
		self.seq_train = from_integer_to_variants(d[2])
		self.split = index

		self.fit()
		self.predict()
		rmsd, _, r2, r2std, pear, spear, hit, f1, ef, ea = self.classic_eval()

		for j, alpha in enumerate(alpha_range):
			coverage[j] = self.get_coverage(alpha=alpha, predictive=False)
			coverage_cons[j] = self.get_coverage(alpha=alpha, predictive=True, measurement_beta=2.)

		print("R2:", r2)

		self.save_predictions(id=str(index), special_identifier=special_identifier)
		if save_model:
			self.save_model(id=str(index), special_identifier=special_identifier)
		self.save_plot(id=str(index), special_identifier=special_identifier)

		return rmsd, r2, r2std, pear, spear, hit, f1, ef, ea, coverage, coverage_cons

	def evaluate_metrics_on_splits(self,split_location = None, no_splits = 10, n_test = 150, output_file = 'output.txt', special_identifier = ''):

		if split_location is None:
			dts = self.cv_split_eval(splits = no_splits, n_test = n_test)
		else:
			dts = self.load_splits_eval(split_location, splits=no_splits)

		splits = no_splits
		rmsds = torch.zeros(size = (splits,1)).view(-1).double()
		coverages = torch.zeros(size = (splits,10)).double()
		coverages_cons = torch.zeros(size = (splits,10)).double()

		r2s = torch.zeros(size = (splits,1)).view(-1).double()
		r2sstd = torch.zeros(size = (splits,1)).view(-1).double()
		pearson = torch.zeros(size = (splits,1)).view(-1).double()
		spearman = torch.zeros(size=(splits, 1)).view(-1).double()

		hit_rates = torch.zeros(size=(splits, 1)).view(-1).double()
		enrichment_factors = torch.zeros(size=(splits, 1)).view(-1).double()
		enrichment_areas = torch.zeros(size=(splits, 1)).view(-1).double()
		f1_scores =  torch.zeros(size=(splits, 1)).view(-1).double()

		alpha_range = np.arange(0.1,1.1,0.1)

		def evaluate(a):
			i,d = a
			return self.evaluate_on_split(i,d,splits = splits,special_identifier=special_identifier)

		results = [evaluate(a) for a in enumerate(dts)]
		for index, v in enumerate(results):
			rmsds[index] = v[0]
			r2s[index] = v[1]
			r2sstd[index] = v[2]
			pearson[index] = v[3]
			spearman[index] = v[4]
			hit_rates[index] = v[5]
			enrichment_factors[index] = v[6]
			enrichment_areas[index] = v[7]
			f1_scores[index] = v[8]
			coverages[index,:] = v[9]
			coverages_cons[index,:] = v[10]

		# output evaluation
		names = ["RMSD","r2","r2std","pearson","spearman","hit_rate","ef","ea","f1"] + ["coverage_"+str(np.round(alpha,1)) for alpha in alpha_range] + ["coverage_cons_"+str(np.round(alpha,1)) for alpha in alpha_range]
		datas = [rmsds,r2s,r2sstd, pearson,spearman, hit_rates,enrichment_factors,enrichment_areas,f1_scores] + [coverages[:,j] for j in range(10)] + [coverages_cons[:,j] for j in range(10)]

		self.save_results(names, datas, output_file = output_file)

	def save_results(self, names, datas, output_file = 'results_strep.txt', file = True):
		with open(self.results_folder+"/"+output_file, 'w') as f:

			print("%20s, %20s: %8s %8s %8s %8s %8s %8s" % ("model name", "quantity", "Q10","Q25", "Q50","Q75", "Q90", "std"))

			for name, data in zip(names, datas):
				print("%20s, %20s: %8f %8f %8f %8f %8f %8f" % (str(self), name, torch.quantile(data, q=0.1),torch.quantile(data, q=0.25),
													   torch.quantile(data, q=0.5),torch.quantile(data, q=0.75), torch.quantile(data, q=0.9),
													   torch.std(data)))
				print("%20s, %20s: %8f %8f %8f %8f %8f %8f" % (str(self), name, torch.quantile(data, q=0.1),torch.quantile(data, q=0.25),
													   torch.quantile(data, q=0.5),torch.quantile(data, q=0.75), torch.quantile(data, q=0.9),
													   torch.std(data)), file=f)
		info = {}
		for name, data in zip(names,datas):
			info[name] = data

		dts = pd.DataFrame(info)
		dts.to_csv(self.results_folder+"/raw"+output_file[0:-4]+".csv")


	def save_plot(self, id = '', special_identifier= '', R2 = False):
		try:
			os.mkdir(self.results_folder + "/plots")
		except:
			print ("plot folder already exists.")

		plt.clf()
		filename = self.results_folder +  "/plots/split_" + special_identifier +id
		if R2:
			R2 = self.get_R2()
			plt.title(str(self)+"_R2:" + str(R2))
		else:
			plt.title("vals")
		plt.xlabel("true")
		plt.ylabel("predicted")
		sigma_std = float(self.estimate_noise_std())
		plt.plot(self.y_test.view(-1).detach().numpy(), self.mu.view(-1).detach().numpy(), color = 'k',marker = 'o', linestyle = '')
		plt.plot(self.y_test,self.y_test,'k-')
		plt.plot(self.y_test, self.y_test+self.s, 'k--')
		plt.plot(self.y_test, self.y_test - self.s, 'k--')

		plt.savefig(filename+ "_0.png", dpi = 150)
		plt.errorbar(self.y_test.view(-1).detach().numpy(), self.mu.view(-1).detach().numpy(), yerr=self.std.view(-1).detach().numpy(),color = 'k',marker = 'o', linestyle = '')

		plt.savefig(filename + "_1.png", dpi = 150)
		plt.errorbar(self.y_test.view(-1).detach().numpy(), self.mu.view(-1).detach().numpy(),yerr=self.std.view(-1).detach().numpy(), xerr= sigma_std+self.std.view(-1).detach().numpy()*0,color = 'r',marker = '', linestyle = '', zorder = -10)
		plt.errorbar(self.y_test.view(-1).detach().numpy(), self.mu.view(-1).detach().numpy(), yerr=self.std.view(-1).detach().numpy(),color = 'k',marker = 'o', linestyle = '')
		plt.savefig(filename + "_2.png", dpi = 150)

		plt.clf()
		#plt.show()

	def save_predictions(self, id = 'None', special_identifier = '', test = True, train = True):
		if test:
			filename = self.results_folder + "/predictions_test_split_"+ id + special_identifier + '.csv'
			dts = pd.DataFrame([self.mu.view(-1).detach().numpy(), self.y_test.view(-1).detach().numpy(), self.std.detach().view(-1).numpy()]).T
			#dts.columns = ["pred", "truth","std"]
			dts.to_csv(filename)

		if train:
			filename = self.results_folder + "/predictions_train_split_"+ id +'.csv'
			dts = pd.DataFrame([self.mu_train.view(-1).detach().numpy(), self.y_test.view(-1).detach().numpy(), self.std_train.detach().view(-1).numpy()]).T
			#dts.columns = ["pred", "truth","std"]
			dts.to_csv(filename)

	def embed(self,x):
		if self.embed_type == "x":
			return self.Embedding.embed(x)
		elif self.embed_type == "seq":
			return self.Embedding.embed_seq(x)

	def fit(self, save_loc=None):
		self.load()
		phi_train = self.embed(self.x_train)

		print ("Fitting with kernel: linear, features", phi_train.size())
		k = KernelFunction(kernel_name="linear", offset = 1., d = phi_train.size()[1])
		self.estimate_noise_std()
		self.GP = GaussianProcess(kernel=k, s = self.s)
		self.GP.fit_gp(phi_train,self.y_train)
		self.fitted = True


	def predict(self, x = None):
		if x is None:
			x = self.x_test
		phi_train = self.embed(self.x_train)
		self.mu_train, self.std_train = self.GP.mean_std(phi_train)
		self.mu, self.std = self.GP.mean_std(self.embed(x))
		return self.mu, self.std

	def get_RMSD(self):
		if self.fitted:
			RMSD = torch.mean((self.mu - self.y_test)**2)
			return RMSD
		else:
			return None

	def get_R2std(self):
		if self.fitted:
			r = r_score_std(self.y_test,self.mu,std = self.s)
			return r
		else:
			return None

	def get_pearson(self):
		if self.fitted:
			r, p = scipy.stats.pearsonr(self.y_test.detach().view(-1).numpy(), self.mu.detach().view(-1).numpy())
			return r
		else:
			return None


	def get_spearman(self):
		if self.fitted:
			r, p = scipy.stats.spearmanr(self.y_test.detach().view(-1).numpy(), self.mu.detach().view(-1).numpy())
			return r
		else:
			return None

	def get_coverage(self, alpha = 0.95, predictive = False, measurement_beta = 2.):
		if self.fitted:
			beta = norm.ppf(alpha)
			if predictive:
				coverage = torch.mean((self.y_test <= self.mu+self.s*measurement_beta+beta*(self.std)).double() * (self.y_test >= self.mu-beta*(self.std)-self.s*measurement_beta).double())
			else:
				coverage = torch.mean((self.y_test <= self.mu+beta*self.std).double() * (self.y_test >= self.mu-beta*self.std).double())
			return coverage
		else:
			return None

	def get_R2(self):
		if self.fitted:
			r2 = sklearn.metrics.r2_score(self.y_test,self.mu)
			return r2
		else:
			return None

	def get_enrichment(self):
		if self.fitted:
			r2 = enrichment_factor(self.y_test, self.mu, alpha = 0.1, threshold=self.threshold)
			return r2
		else:
			return None

	def get_enrichment_area(self):
		if self.fitted:
			r2 = enrichment_area(self.y_test, self.mu, alpha=0.1)
			return r2
		else:
			return None


	def get_f1(self):
		if self.fitted:
			r2 = f1_score(self.y_test, self.mu, threshold=self.threshold)
			return r2
		else:
			return None


	def get_hit_rate(self):
		if self.fitted:
			r2 = hit_rate(self.y_test, self.mu, threshold=self.threshold)
			return r2
		else:
			return None