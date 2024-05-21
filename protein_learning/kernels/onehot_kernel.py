import numpy as np
from scipy.spatial.distance import hamming
import torch
from collections import Counter

def onehot_kernel(seqs1, seqs2):
    """
    Returns a matrix of hamming distances between all pairs of sequences in seqs1 and seqs2
    """
    return torch.from_numpy(np.array([[ sum(min(Counter(seq1)[char], Counter(seq2)[char]) for char in Counter(seq1) if char in Counter(seq2))
 for seq2 in seqs2] for seq1 in seqs1]))

if __name__ == "__main__":
    seqs1 = ['KK','AA','SS']
    seqs2 = ['AB','SQ']

    print (onehot_kernel(seqs1,seqs2))
    print (onehot_kernel(seqs1,seqs1))