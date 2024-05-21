from mutedpy.protein_learning.embeddings.amino_acid_embedding import AminoAcidEmbedding
from mutedpy.protein_learning.embeddings.pair_amino_acid_embedding import PairAminoAcidEmbedding

import os
import numpy as np
import torch

def test_embedding():
	if os.path.dirname(__file__) == "":
		path_dir = "./"
	else:
		path_dir = os.path.dirname(__file__)

	embedding1 = AminoAcidEmbedding(path_dir+"../../experiments/data/amino-acid-features.csv", n_sites = 4)
	embedding2 = AminoAcidEmbedding(path_dir+"../../experiments/data/amino-acid-features-full.csv", n_sites = 4)
	embedding3 = PairAminoAcidEmbedding(path_dir+"../../experiments/data/amino-acid-pair-features-full-1.csv", n_sites = 4, pairs = None)
	embedding4 = PairAminoAcidEmbedding(path_dir+"../../experiments/data/amino-acid-pair-features-full-2.csv", n_sites = 4, pairs = None)


	variant = torch.Tensor([[0,1,1,0]])

	print (embedding1.embed(variant).size())
	print (embedding2.embed(variant).size())
	print (embedding3.embed(variant).size())
	print (embedding4.embed(variant).size())

if __name__ == "__main__":
	test_embedding()