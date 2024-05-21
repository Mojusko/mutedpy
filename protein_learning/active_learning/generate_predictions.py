import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from stpy.continuous_processes.gauss_procc import GaussianProcess
from mutedpy.protein_learning.embeddings.amino_acid_embedding import AminoAcidEmbedding

def load_model(params,log_name = 'none', model_params_return=False, change_database = None, vintage = False, ems = False):

	model_params = pickle.load(open(params, "rb"))
	k = model_params['kernel_object']
	phi = model_params['x']
	yy = model_params['y']

	s = model_params['noise_std']
	if vintage:
		print (model_params.keys())
	else:
		feature_loader = model_params['feature_loader']
	feature_mask = model_params['feature_mask']
	GP = GaussianProcess(kernel=k, s=s)
	print("Fitting", log_name)
	print(k.params)
	if 'Sigma' in model_params.keys():
		GP.fit_gp(phi, yy, Sigma = model_params['Sigma'])
	else:
		GP.fit_gp(phi, yy)
	if change_database is not None:
		feature_loader.feature_loaders[0].Embedding.project = change_database
	if vintage:
		path_dir = os.path.dirname(__file__)
		feature_loader = AminoAcidEmbedding(data=path_dir + "/../../experiments/streptavidin/data/amino-acid-features.csv",
											 projection=path_dir + "/../../experiments/streptavidin/data/embedding-dim5-demean-norm.pt",
											 proto_names=5)
		embed = lambda x: feature_loader.embed(x)[:,feature_mask]
	elif ems:
		embed = lambda x: feature_loader.embed(x)[:, feature_mask]

	else:
		feature_loader.connect()
		embed = lambda x: feature_loader.embed(x)[:, feature_mask]

	if not model_params_return:
		return GP,embed

	else:
		return GP, embed, model_params