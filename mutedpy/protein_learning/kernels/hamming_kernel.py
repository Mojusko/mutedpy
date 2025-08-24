import numpy as np
from scipy.spatial.distance import hamming
import torch
def hamming_kernel(seqs1, seqs2):
    """
    Returns a matrix of hamming distances between all pairs of sequences in seqs1 and seqs2
    """
    return torch.from_numpy(np.array([[1./(1+hamming(list(seq1), list(seq2)) * len(seq1)) for seq2 in seqs2] for seq1 in seqs1]))

if __name__ == "__main__":
    seqs1 = ['KK','AA','SS']
    seqs2 = ['AB','SQ']

    print (hamming_kernel(seqs1,seqs2))
    print (hamming_kernel(seqs1,seqs1))