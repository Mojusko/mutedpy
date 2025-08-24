import pandas as pd

from mutedpy.experiments.od1.loader import load_data_pickle_process, get_wiltype_seq
from mutedpy.utils.sequences.sequence_utils import from_variant_to_integer, from_variants_to_mutations
import torch

def mutability_scores(seq, y, wt, median = False):
    n = len(seq[0])
    x = from_variant_to_integer(seq)
    x_wt = from_variant_to_integer(wt)
    mutability = torch.zeros(n).double()
    lmutability = torch.zeros(n).double()
    rmutability = torch.zeros(n).double()
    for i in range(n):
        mask = x[:,i] != x_wt[0,i]
        if median:
            mutability[i] = torch.median(y[mask])
        else:
            mutability[i] = torch.mean(y[mask])
        lmutability[i] =torch.quantile(y[mask], 0.25)
        rmutability[i] = torch.quantile(y[mask], 0.75)

    return mutability, lmutability, rmutability
def combinability_scores(seq, y, wt):
    n = len(seq[0])
    m = len(seq)
    combinability_val = torch.zeros(n).double()

    x = from_variant_to_integer(seq)
    x_wt = from_variant_to_integer(wt)

    hamming_distance = torch.Tensor([float(torch.sum(a != x_wt, dim = 1)) for a in x])
    active_indicator_mask = y > 0
    mutations = from_variants_to_mutations(seq, wt, range(n))
    dts = pd.DataFrame(data = {'seq':seq,'Mutation':mutations,'y':y})
    dictionary = dts.set_index('Mutation')['y'].to_dict()

    for i in range(n):
        mask = x[:,i] != x_wt[0,i]
        epistatic_mask = torch.zeros(m).bool()
        for j in range(m):
            ham = hamming_distance[j]
            if mask[j] == True and ham > 1:
                additive_value = 0.
                for mut in mutations[j].split("+"):
                    if mut in dictionary.keys():
                        additive_value+=dictionary[mut]
                    else:
                        pass
                epistatic_mask[j] = True if additive_value < dictionary[mutations[j]] else False
        combinability_val[i] = torch.sum(hamming_distance[active_indicator_mask & epistatic_mask])
    return combinability_val

def simple_combinability_scores(seq, y, wt):
    n = len(seq[0])
    m = len(seq)
    combinability_val = torch.zeros(n).double()

    x = from_variant_to_integer(seq)
    x_wt = from_variant_to_integer(wt)

    hamming_distance = torch.Tensor([float(torch.sum(a != x_wt, dim=1)) for a in x])
    ones = torch.ones(m)
    active_indicator_mask = y > 0

    for i in range(n):
        mask = x[:, i] != x_wt[0, i]
        epistatic_mask = torch.zeros(m).bool()
        for j in range(m):
            ham = hamming_distance[j]
            if mask[j] == True and ham > 1:
                epistatic_mask[j] = 1.
        combinability_val[i] = torch.sum(ones[active_indicator_mask & epistatic_mask])
    return combinability_val


if __name__ == "__main__":
    wt_seq = get_wiltype_seq()
    x,y,seq = load_data_pickle_process()
    y_real = y.int().sum(dim = 1)
    import matplotlib.pyplot as plt

    combinability_score = combinability_scores(seq, torch.log(y_real.float()), wt_seq)
    simple_combinability_score = simple_combinability_scores(seq, torch.log(y_real.float()), wt_seq)

    plt.bar(range(len(wt_seq)),combinability_score, color = 'tab:blue')
    plt.figure()
    plt.bar(range(len(wt_seq)),simple_combinability_score, color = 'tab:red')
    # mutability, l, r = mutability_scores(seq, y_real.float(), wt_seq)
    # plt.bar(range(len(wt_seq)),mutability)
    plt.xlabel('Position', fontsize=12)
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed

    plt.show()