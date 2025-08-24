import pandas as pd
import torch
from Levenshtein import distance

def mask_filter(list_of_seqs):
    output = []
    indices = []
    for index, s in enumerate(list_of_seqs):
         if "<mask>" not in s:
             output.append(s)
             indices.append(index)
    return output,indices

# def uniqueness_filter(list_of_seqs):
#     new_seq = list(dict.fromkeys(list_of_seqs))
#     boolean_mask = [list_of_seqs.index(element) == i for i, element in enumerate(list_of_seqs)]
#     return new_seq, boolean_mask

def remove_wildtype_filter(list_of_seqs, wildtype):
    output = []
    indices = []
    for index, s in enumerate(list_of_seqs):
        if wildtype != s:
            output.append(s)
            indices.append(index)
    return output, indices

def sorting_filter(list_of_seq, y, unique = True):
    new_seq = []
    new_y = []
    no_sorts = y.size()[1]
    new_seqs = pd.DataFrame(list_of_seq, columns=['variant'])

    for i in range(no_sorts):
        new_seqs[str(i)] = y[:, i]
    new_seqs['occ'] = 1.

    agg_dict = {}
    agg_list = [{str(i) : 'sum'} for i in range(no_sorts)] + [{'occ':'sum'}]
    for d in agg_list:
        agg_dict.update(d)

    new_seqs = new_seqs.groupby('variant').agg(agg_dict).reset_index()

    # for i in range(no_sorts):
    #     new_seqs[str(i)] = new_seqs[str(i)].apply(lambda x: min([x,1]))

    def is_sorted_descending(lst):
        return all(lst[i] > lst[i + 1] or (lst[i+1]==0 and lst[i]==0) for i in range(len(lst) - 1))

    for index, row in new_seqs.iterrows():

        if is_sorted_descending([row[str(i)] for i in range(no_sorts)]):
           # print ("good.", row)

            yn = torch.zeros(size=(1, no_sorts)).bool()
            if unique:
                new_seq.append(row['variant'])
                ac = max([i if row[str(i)] > 0 else 0 for i in range(no_sorts)])
                yn[0, 0:ac + 1] = 1.
                new_y.append(yn)
            else:
                ac = max([i if row[str(i)] > 0 else 0 for i in range(no_sorts)])
                yn[0, 0:ac + 1] = 1.
                for _ in range(int(row['occ'])):
                    new_seq.append(row['variant'])
                    new_y.append(yn)
    # # unique sequences
    # unique_seqs = list(set(new_seqs.values.tolist()))
    #
    # for index, seq in enumerate(unique_seqs):
    #     print (index, len(unique_seqs))
    #     mask = new_seqs == seq
    #     entries = y[mask, :]
    #
    #     occurences = entries.size()[0]
    #     #print (entries)
    #     #print (entries.sum(dim = 0))
    #     sums = entries.sum(dim = 0)
    #     if all([sums[i] == max(0,occurences-i) for i in range(no_sorts)]):
    #         # print ('------------------')
    #         # print("true entry.")
    #         # print('------------------')
    #         for i in range(occurences):
    #             new_seq.append(seq)
    #         new_y.append(entries)
    #

    new_y = torch.vstack(new_y)
    print ("Filtering finished:")
    for i in range(3):
        print (torch.sum(new_y[:, i]))

    return new_seq, new_y

def last_filter_seq(s):
    if s[-1] == ">":
       return s[0:-6]
    else:
        return s[0:-1]
def last_filter(list_of_seqs):
    new_seqs = []
    indices = []
    for index, s in enumerate(list_of_seqs):
        # remove starting and ending
        new_seqs.append(last_filter_seq(s))
        indices.append(index)
    return new_seqs, indices

def initial_last_filter(list_of_seqs):
    new_seqs = []
    indices = []
    for index, s in enumerate(list_of_seqs):
        # remove starting and ending
        if s[0]=="<" and s[-1]==">":
            new_seqs.append(s[6:-6])
            indices.append(index)
        # remove starting
        if s[0]=="<" and s[-1]!=">":
            new_seqs.append(s[6:-1])
            indices.append(index)
        # remove ending
        if s[0]!="<" and s[-1]==">":
            new_seqs.append(s[1:-6])
            indices.append(index)
        # remove starting and ending correct
        if s[0]!="<" and s[-1]!=">":
            new_seqs.append(s[1:-1])
            indices.append(index)
    return new_seqs, indices


def initial_last_filter_seq(s):
    if s[0] == "<" and s[-1] == ">":
        return s[6:-6]
    # remove starting
    if s[0] == "<" and s[-1] != ">":
        return s[6:-1]
    # remove ending
    if s[0] != "<" and s[-1] == ">":
        return s[1:-6]
    # remove starting and ending correct
    if s[0] != "<" and s[-1] != ">":
        return s[1:-1]

def number_of_mutations_filter(list_of_seqs, wildype_seq, cutoff):
    output = []
    indices = []
    for index, s in enumerate(list_of_seqs):
        d = distance(s,wildype_seq)
        if d<cutoff:
            output.append(s)
            indices.append(index)
    return output,indices


if __name__ == "__main__":
    seq = ['AB','AA','AB','BB','CC','AA']
    new_seq, indices = uniqueness_filter(seq)
    print (seq)
    print (new_seq)
    print (indices)
    print (seq[indices])
