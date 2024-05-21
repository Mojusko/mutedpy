import numpy as np
import stpy.helpers.helper as helper

class ProteinOperator():

    def __init__(self):

        self.real_names = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys', 'Q': 'Gln', 'E': 'Glu',
                           'G': 'Gly',
                           'H': 'His', 'I': 'Ile', 'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe',
                           'P': 'Pro', 'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val', 'B': 'Asx'}

        self.real_names_cap = {k: v.upper() for k, v in self.real_names.items()}

        self.dictionary = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
                           'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13,
                           'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'B': 3}

    
        self.pair_dictionary = {}
        for index, key in enumerate(self.dictionary.keys()):
            size = len(self.dictionary.keys())-1
            for index2, key2 in enumerate(self.dictionary.keys()):
                self.pair_dictionary[key+key2] = self.dictionary[key]*size + self.dictionary[key2]

        self.inv_dictionary = {v: k for k, v in self.dictionary.items()}
        self.inv_dictionary[3] = 'D'
        self.inv_real_names = {v: k for k, v in self.real_names.items()}
        self.inv_real_names_cap = {v: k for k, v in self.real_names_cap.items()}

        self.Negative = ['D', 'E']
        self.Positive = ['R', 'K', 'H']
        self.Aromatic = ['F', 'W', 'Y', 'H']
        self.Polar = ['N', 'Q', 'S', 'T', 'Y']
        self.Aliphatic = ['A', 'G', 'I', 'L', 'V']
        self.Amide = ['N', 'Q']
        self.Sulfur = ['C', 'M']
        self.Hydroxil = ['S', 'T']
        self.Small = ['A', 'S', 'T', 'P', 'G', 'V']
        self.Medium = ['M', 'L', 'I', 'C', 'N', 'Q', 'K', 'D', 'E']
        self.Large = ['R', 'H', 'W', 'F', 'Y']
        self.Hydro = ['M', 'L', 'I', 'V', 'A']
        self.Cyclic = ['P']
        self.Random = ['F', 'W', 'L', 'S', 'D']

    def translate(self, X):
        f = lambda x: self.dictionary[x]
        Y = np.zeros(shape=X.shape).astype(int)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Y[i, j] = f(X[i, j])
        return Y

    def remove_wild_type_mutations(self, mutation):
        mutation_split = mutation.split("+")
        output = []
        for mut in mutation_split:
            if mut[0] != mut[-1]:
                output.append(mut)
        return "+".join(output)

    def get_variant_code(self, mutation):
        mutation_split = mutation.split("+")
        return "".join([mut[-1] for mut in mutation_split])

    def get_substitutes_from_mutation(self, mutation):
        mutation_split = mutation.split("+")
        original = []
        new = []
        positions = []

        for mut in mutation_split:
            original.append(mut[0])
            new.append(mut[-1])
            positions.append(int(mut[1:-1]))

        return (original, new, positions)

    def mutation(self,
                 original_seq,
                 positions,
                 new_seq,
                 neutral = True):
        old_seq = list(original_seq)
        new_seq = list(new_seq)
        if neutral:
            identifier = [old + str(position) + new for old, new, position in zip(old_seq, new_seq, positions)]
        else:
            identifier = [old + str(position) + new for old, new, position in zip(old_seq, new_seq, positions) if old!=new]
        return '+'.join(identifier)

    def interval_number(self, dim=None):
        if dim is None:
            dim = self.dim
        arr = self.interval_letters(dim=dim)
        out = self.translate(arr)
        return out

    def interval_onehot(self, dim=None):
        if dim is None:
            dim = self.dim
        arr = self.interval_letters(dim=dim)
        out = self.translate_one_hot(arr)
        return out

    def interval_letters(self, dim=None):
        if dim is None:
            dim = self.dim

        names = list(self.dictionary.keys())
        names.remove('B')
        arr = []
        for i in range(dim):
            arr.append(names)
        out = helper.cartesian(arr)
        return out

    def translate_amino_acid(self, letter):
        return self.dictionary[letter]

    def translate_seq_to_variant(self, sec):
        return "".join([self.inv_dictionary[int(s)] for s in sec])

    def translate_mutation_variant(self, series):
        f = lambda x: np.array(list(map(int, [self.inv_dictionary[a] for a in list(str(x))]))).reshape(-1, 1)
        xtest = np.concatenate(series.apply(f).values, axis=1).T
        return xtest

    def translate_mutation_series(self, series):
        f = lambda x: np.array(list(map(int, [self.dictionary[a] for a in list(str(x))]))).reshape(-1, 1)
        xtest = np.concatenate(series.apply(f).values, axis=1).T
        return xtest

    def translate_one_hot(self, X):
        try:
            Y = self.translate(X)
        except:
            Y = X
        n, d = list(X.shape)
        Z = np.zeros(shape=(n, d * self.total))
        for i in range(n):
            for j in range(d):
                Z[i, Y[i, j] + j * self.total] = 1.0

        return Z

    def get_real_name(self, name):
        out = []
        for i in name:
            out.append(self.real_names[i])
        return out
