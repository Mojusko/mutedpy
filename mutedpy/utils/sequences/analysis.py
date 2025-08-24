import pandas as pd
import torch
import numpy as np
from collections import Counter


def analyze_frequencies(seqs):
    d = [Counter(position) for position in zip(*seqs)]
    return [len(list(d[i].keys())) for i in range(len(d))]

def analyze_frequencies_aa(seqs):
    d = [Counter(position) for position in zip(*seqs)]
    return d

def diff_counts(baseline, string_list):
    # Transpose the strings
    transposed = list(zip(*string_list))
    # Count the occurrences of each character at each position
    counts = [Counter(s) for s in transposed]
    # Calculate the number of differences at each position
    diff_counts = [sum(c.values()) - c[baseline[i]] for i, c in enumerate(counts)]
    return diff_counts


if __name__ == "__main__":
    strings = ["abc", "axc", "ayz"]
    d = diff_counts("axa",strings)
    print (d)
    # print([len(list(d[i].keys())) for i in range(len(d))])
