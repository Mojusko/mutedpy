import blosum as bl
import numpy as np
import torch
def blosum_kernel(x, y):
        """
        Return score for a given pair of residues in a give matrix.

        Matrices only have one triangle, so if res1 -> res2 isn't in matrix
        res2 -> res1 will be...
        """

        # load blosum matrix
        matrix = bl.BLOSUM(90)

        K = np.zeros((len(x), len(y)))

        # Nested loop for each pair of sequences in x and y
        for i, seq_x in enumerate(x):
            for j, seq_y in enumerate(y):
                # Vectorized computation of scores for each pair of amino acids in the sequences
                K[i, j] = sum(matrix[aa_x][aa_y] for aa_x, aa_y in zip(seq_x, seq_y))

        return torch.from_numpy(K)


if __name__ == "__main__":
    seqs1 = ['KK','AA','SS']
    seqs2 = ['AB','SQ']

    print (blosum_kernel(seqs1,seqs2))

    print (blosum_kernel(seqs1,seqs1))