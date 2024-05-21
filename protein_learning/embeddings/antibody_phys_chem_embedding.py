import pandas as pd
import numpy as np
import torch
from mutedpy.utils.protein_operator import ProteinOperator
from mutedpy.protein_learning.featurizers.feature_loader import ProteinFeatureLoader
from mutedpy.protein_learning.embeddings.amino_acid_embedding import AminoAcidEmbedding
import stpy.helpers.helper as helper
"""
Implemented from paper by ... 
"""

class AntibodyPhysChemEmbedding(AminoAcidEmbedding):

	def __init__(self, data = 'PI.csv'):
		self.alph = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
		self.m = 26

	def embed(self, x):
		# TODO: Implement in our notation and feature information using numbers instead of letters
		n, no_sites = x.size()
		res_counts = pd.DataFrame(index=self.alph)

		for i in range(n):


			hydrophobicity = []

			for column in res_counts:
				hydros = []
				for index, row in res_counts.iterrows():
					hydros.append(row[column] * residue_info.loc[column, 'Hydropathy Score'])
				hydrophobicity.append(hydros)

		hydrophobicity = pd.DataFrame(hydrophobicity).T
		hydrophobicity['ave'] = hydrophobicity.sum(axis=1) / 115
		res_counts['Hydro'] = res_counts['A'] + res_counts['I'] + res_counts['L'] + res_counts['F'] + res_counts[
			'V']
		res_counts['Amph'] = res_counts['W'] + res_counts['Y'] + res_counts['M']
		res_counts['Polar'] = res_counts['Q'] + res_counts['N'] + res_counts['S'] + res_counts['T'] + res_counts[
			'C'] + \
							  res_counts['M']
		res_counts['Charged'] = res_counts['R'] + res_counts['K'] + res_counts['D'] + res_counts['E'] + res_counts[
			'H']

		return physchemvh