from mutedpy.protein_learning.embeddings.amino_acid_embedding import AminoAcidEmbedding
from stpy.helpers.helper import generate_all_pairs
from mutedpy.utils.protein_operator import ProteinOperator
import pandas as pd
import torch

class PairAminoAcidEmbedding(AminoAcidEmbedding):

	def __init__(self, data = 'pair-amino-acid-features.csv', n_sites = 5, pairs = None):
		super().__init__(data = data, n_sites = n_sites)


		if pairs is None:
			self.pairs = generate_all_pairs(n_sites)
		else:
			self.pairs = pairs
	def load(self):

		dts = pd.read_csv(self.datafile, keep_default_na=False)
		self.Opt = ProteinOperator()
		self.dict = {}

		for index, row in dts.iterrows():
			pair = row['aminoacid']
			vec = torch.from_numpy(row[1:].values.astype(float))
			self.dict[self.Opt.pair_dictionary[pair]] = vec

		self.projected_components = vec.size()[0]*self.n_sites
		self.m = self.projected_components
		self.tol = 10e-6
		self.feature_names = dts.columns[1:]


	def embed(self, x):
		n,no_sites = x.size()
		vec_per_pair = []
		for pair in self.pairs:
			i = pair[0]
			k = pair[1]
			vec = torch.cat([self.dict[int(x[j,i])*20+int(x[j,k])].view(1,-1) for j in range(n)])
			vec_per_pair.append(vec)
		vec = torch.hstack(vec_per_pair)

		if self.projected == True:
			if self.mean is not None:
				vec =  (vec - torch.tile(self.mean, (vec.size()[0],1)))@self.P
			else:
				vec = vec@self.P
		else:
			if self.mean is not None:
				 vec = vec - torch.tile(self.mean, (vec.size()[0],1))
		return vec