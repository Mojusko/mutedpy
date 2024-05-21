import scipy
import numpy as np
import torch
from tinydb import TinyDB
import pandas as pd
from mutedpy.utils.protein_operator import ProteinOperator
from stpy.helpers.helper import cartesian
from Levenshtein import distance
from textdistance import levenshtein

def get_sites(series):
	"""
		get all mutation sites
	"""
	d = {}
	for elem in series:
		try:
			for mut in elem.split('+'):
				e = int(mut[1:-1])
				try:
					d[e] = d[e] + 1
				except:
					d[e] = 1
		except:
			pass
	return list(d)

def shift_mutations(mutation,offset):
	mutation_split = mutation.split("+")
	try:
		output = []
		for mut in mutation_split:
			pos = int(mut[1:-1]) - offset
			output.append(mut[0] + str(pos) + mut[-1])
		return '+'.join(output)
	except:
		return mutation

def from_mutation_to_variant(list):
	func = lambda x: "".join(p[-1] for p in x.split("+"))
	return [func(a) for a in list]

def from_variant_to_integer(list):
	Op = ProteinOperator()
	x = torch.from_numpy(Op.translate_mutation_series(pd.Series(list)))
	return x

def from_mutation_to_variant_custom_parent(seqs, parent):
	output = []
	for s in seqs:
		new_parent = list(parent)
		for mut in s.split("+"):
			new_parent[int(mut[1:-1])] = mut[-1]
		output.append("".join(new_parent))
	return output
def add_variant_column(dataframe, name = 'Mutation'):
	func = lambda x: "".join(p[-1] for p in x.split("+"))
	dataframe['variant'] = dataframe[name].apply(func)
	return dataframe

def drop_neural_mutations(dataframe):
	check_neutral = lambda a: a if a[0] != a[-1] else ""
	func = lambda x: "+".join([check_neutral(a) for a in x.split("+") if check_neutral(a)!=""])
	dataframe['Mutation'] = dataframe['Mutation'].apply(func)
	return dataframe

def create_neural_mutations(dataframe):
	check_neutral = lambda a: a if a[0] != a[-1] else ""
	func = lambda x: "+".join([check_neutral(a) for a in x.split("+") if check_neutral(a)!=""])
	dataframe['Mutation_n'] = dataframe['Mutation'].apply(func)
	return dataframe

def shift_mutation_string(x, shift = 0):
	y = list(map(lambda x: x[0]+str(int(x[1:-1])+shift)+x[-1], x.split("+")))
	return "+".join(y)

def shift_mutations(dataframe, shift = 0):
	for index, row in dataframe.iterrows():
		if row['Mutation'] != "":
			x = row['Mutation']
			dataframe['Mutation'][index] = shift_mutation_string(x, shift = shift)
		try:
			if row['control units'] != "":
				x = row['control units']
				dataframe['control units'][index] = shift_mutation_string(x, shift=shift)
		except:
			pass
	return dataframe


def order_mutation_string(x):
	if x!="":
		y = list(map(lambda x: int(x[1:-1]), x.split("+")))
		return "+".join(list(np.array(x.split("+"))[np.argsort(y)]))
	else:
		return ""
def order_mutations(dataframe):
#	counter = 0
	dataframe['Mutation'] = dataframe['Mutation'].apply(order_mutation_string)
	# for index, row in dataframe.iterrows():
	# 	print (counter, len(dataframe))
	# 	if row['Mutation'] != "":
	# 		x = row['Mutation']
	# 		dataframe['Mutation'][index] = order_mutation_string(x)
	# 	counter+=1
	return dataframe


def remove_unspecific_mutations(dataframe):
	newdataframe = dataframe.copy()
	for index, row in dataframe.iterrows():
		if row['Mutation'] != "":
			x = row['Mutation'].split("+")
			letters = "".join([a[0]+a[-1] for a in x])
			if not letters.isalpha():
				newdataframe = dataframe.drop(index, axis = 0)
	return newdataframe

def number_of_muts(mutant):
	return len(mutant.split("+"))

def get_number_of_muts(dts):
	return dts['Mutation'].apply(number_of_muts)

def change_fasta(fasta, mutation):
	"""
		 add mutation to FASTA
	"""
	fasta_list = list(fasta)
	mutation_split = mutation.split("+")
	try:
		for mut in mutation_split:
			pos = int(mut[1:-1])-1
			fasta_list[pos] = mut[-1]
		return ''.join(fasta_list)
	except:
		print ("Failing...")
		return fasta

def sample_mutation(sites,parent):
	Op = ProteinOperator()
	aas = list(Op.dictionary.keys())
	aas.remove('B')
	d = len(sites)
	muts = []
	for i in range(d):
		k = int(np.random.randint(0,len(aas),1))
		muts.append(parent[i] + str(sites[i]) + aas[k])
	return "+".join(muts)

def generate_random_mutations(n, sites, parent,prior_muts = None):
	mutations = []
	i = 0
	while i < n:
		mutation = sample_mutation(sites,parent)
		if prior_muts is None:
			i = i + 1
			mutations.append(mutation)
		else:
			if np.sum((prior_muts == mutation).values)>0:
				pass
			else:
				i = i + 1
				mutations.append(mutation)
	return mutations

def generate_random_variants_limited(n, sites, parents, dist = 5):
	Op = ProteinOperator()
	aas = list(Op.dictionary.keys())
	aas.remove('B')
	output = []
	for _ in range(n):
		mutation = list(parents)
		text_output = ""
		for j in range(dist):
			# sample random site
			site = int(np.random.randint(0,len(sites),1))
			# sample random letter from
			mutation[sites[site]] = aas[int(np.random.randint(0,len(aas),1))]
			text_output+=str(sites[site])+aas[int(np.random.randint(0,len(aas),1))]
		print ("mutating", text_output, distance(mutation, parents))
		output.append("".join(mutation))
	return output
def generate_all_combination(sites, parent):
	Op = ProteinOperator()
	aas = list(Op.dictionary.keys())
	aas.remove('B')
	d = len(sites)
	mutations = cartesian([aas for _ in range(d)])
	out = []
	for mut in mutations:
		out.append("+".join([parent[i] + str(sites[i]) + mut[i] for i in range(d)]))
	return out

def hamming_distance(mutation, prior_muts):

	def hamming(s1,s2):
		return sum(c1 != c2 for c1, c2 in zip(s1, s2))

	ham = lambda x: hamming(mutation,x)
	values = prior_muts['variant'].apply(ham)
	return np.min(values.values)

def add_hamming_distance(dts, reference):
	ham = dts.apply(lambda x: levenshtein.distance(x['variant'], reference), axis=1)
	dts['hamming'] = ham
	return dts


def add_mutation_column(dts, positions, wildtype, neutral = True):
	Op = ProteinOperator()
	f = lambda x:  Op.mutation(wildtype, positions, x, neutral=neutral)
	dts['Mutation'] = dts['variant'].apply(f)
	return dts

def from_variants_to_mutations(seq, parent,positions, neutral = False):
	Op = ProteinOperator()
	f = lambda x: Op.mutation(parent, positions, x, neutral=neutral)
	new_seq = [f(s) for s in seq]
	return new_seq

def filter_by_amino_acid_validity(dts):
	Op = ProteinOperator()
	f = lambda x: all([a in Op.dictionary.keys() for a in x])
	mask = dts['variant'].apply(f)
	dts = dts[mask]
	return dts

def from_integer_to_variants(tensor):
	n, d = tensor.size()
	Op = ProteinOperator()
	vecs = []
	for i in range(d):
		vecs.append(pd.Series(tensor[:,i].numpy()).apply(lambda x: Op.inv_dictionary[x]).values.tolist())
	variants = []
	for i in range(n):
		s = ''
		for j in range(d):
			s = s+ vecs[j][i]
		variants.append(s)
	return variants

def translate_long_aa_names_to_short_names(list, capitals = True):
	Op = ProteinOperator()
	out = []
	for aa in list:
		if capitals:
			out.append(Op.inv_real_names_cap[aa])
		else:
			out.append(Op.inv_real_names[aa])
	return out













