import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

class ProteinOperator():

	def __init__(self):
		"""
			 text manipulation with amino acid names
		"""
		self.real_names = {'A':'Ala', 'R':'Arg', 'N':'Asn', 'D':'Asp', 'C':'Cys','Q':'Gln',  'E':'Glu','G':'Gly',
				'H':'His','I':'Iso','L':'Leu',	'K':'Lys','M':'Met','F':'Phe',
				'P':'Pro','S':'Ser','T':'Thr','W':'Trp','Y':'Tyr','V':'Val','B':'Asx'}

		self.dictionary = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4,'Q':5,'E':6,'G':7,
				'H':8,'I':9,'L':10,	'K':11,'M':12,'F':13,
				'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19,'B':3}


		self.Negative = ['D', 'E']
		self.Positive = ['R', 'K', 'H']
		self.Aromatic = ['F', 'W', 'Y','H']
		self.Polar = ['N', 'Q', 'S', 'T','Y']
		self.Aliphatic = ['A','G','I','L','V']
		self.Amide = ['N','Q']
		self.Sulfur = ['C','M']
		self.Hydroxil = ['S','T']
		self.Small = ['A', 'S', 'T', 'P', 'G', 'V']
		self.Medium = ['M', 'L', 'I', 'C', 'N', 'Q', 'K', 'D', 'E']
		self.Large = ['R', 'H', 'W', 'F', 'Y']
		self.Hydro = ['M', 'L', 'I', 'V', 'A']
		self.Cyclic = ['P']
		self.Random = ['F', 'W', 'L', 'S', 'D']


	def translate(self,X):
		"""
			translate letter -> number
		"""
		f = lambda x: self.dictionary[x]
		Y = X.copy()
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				Y[i,j] = f(X[i,j])
		return Y.astype(int)


	def translate_amino_acid(self,letter):
		"""
		letter -> number
		"""
		return self.dictionary[letter]

	def translate_one_hot(self,X):
		"""
			generate one-hote encoding
		"""
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

	def get_real_name(self, name):
		"""
			get longer name
		"""
		out = []
		for i in name:
			out.append(self.real_names[i])
		return out


