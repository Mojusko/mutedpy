import pandas as pd
import numpy as np
import torch
from mutedpy.utils.protein_operator import ProteinOperator
# from sklearn.neighbors import KernelDensity
# from scipy.signal import argrelextrema
from mutedpy.protein_learning.featurizers.feature_loader import ProteinFeatureLoader
from mutedpy.utils.sequences.sequence_utils import from_variant_to_integer
import stpy.helpers.helper as helper


class AminoAcidEmbedding(ProteinFeatureLoader):

    def __init__(self, data='amino-acid-features.csv',
                 n_sites=5,
                 projection=None,
                 proto_names=None,
                 demean = None,
                 stacking = False,
                 append = 0):
        # create a lookup
        self.datatype = 'internal'
        self.mean = None
        self.stacking = stacking
        self.splits = None
        self.append = append
        self.projected = False
        print("Amino-acids embedding loaded.")
        self.datafile = data
        self.n_sites = n_sites
        self.load()

        if projection is not None:
            self.load_projection(projection)
            if proto_names is not None:
                self.set_proto_names(proto_names)
        if demean is not None:
            self.demean(n_sites, local = demean)

    def set_positions(self, positions):
        self.positions = positions

    def load(self):
        dts = pd.read_csv(self.datafile)
        self.Opt = ProteinOperator()
        self.dict = {}
        for index, row in dts.iterrows():
            aminoacid = row['aminoacid']
            vec = torch.from_numpy(row[1:].values.astype(float))
            self.dict[self.Opt.dictionary[aminoacid]] = vec
        self.projected_components = vec.size()[0] * self.n_sites
        self.m = self.projected_components
        self.tol = 10e-6
        self.feature_names = [a + 'site_' + str(site) for a in dts.columns[1:].values.tolist() for site in
                              range(self.n_sites)]

    def split_features(self, xtest=None, bandwidth=0.2, demean=True, normalize=True, n_sites=5):
        xtest = torch.from_numpy(self.Opt.interval_number(dim=n_sites))

        N = 10000
        # subsample
        xtest = xtest[np.random.randint(0, xtest.size()[0], N), :]
        embedding = self.embed(xtest)

        n, no_features = embedding.size()
        self.splits = {}
        newy = embedding.clone()
        for i in range(no_features):
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(embedding[:, i].numpy().reshape(-1, 1))
            xmin = torch.min(embedding[:, i])
            xmax = torch.max(embedding[:, i])
            x_plot = torch.linspace(xmin, xmax, 1024)
            log_dens = kde.score_samples(x_plot.view(-1, 1))
            breakpoints = argrelextrema(log_dens, np.less)
            splits = []
            if len(breakpoints[0]) > 0:
                for index, b in enumerate(breakpoints[0]):
                    print("breakpoint detected in feature:", i, "index:", index)
                    splits.append(x_plot[b])
                    mask = embedding[:, i] > float(x_plot[b])
                    ynew1 = torch.zeros(size=(embedding.size()[0], 1)).double()
                    ynew1[mask] = embedding[mask, i].view(-1, 1)
                    ynew2 = torch.zeros(size=(embedding.size()[0], 1)).double()
                    ynew2[~mask] = embedding[~mask, i].view(-1, 1)
                    newy[:, i] = ynew1.view(-1)
                    newy = torch.hstack((newy, ynew2))
                    self.feature_names.append(self.feature_names[i] + "_splitted_" + str(index))
            else:
                print("no break point in ", i)
            self.splits[i] = splits

        self.m = len(self.feature_names)

        self.mean_split = torch.mean(newy, dim=0)

        norm = torch.max(torch.abs(newy), dim=0)[0]
        self.P_split = torch.diag(1. / norm)

    # def embed_seq(self, x):
    #     if not hasattr(self, "positions"):
    #         raise ModuleNotFoundError("You have to first set positions attribute")
    #     else:
    #         # convert selected to integers
    #         int_sites_preprocess = [[int(a[1:-1]) for a in mut.split("+")] for mut in x]
    #         variants = [[a[-1] for a in mut.split("+")] for mut in x]
    #         out = []
    #         for index1, sites in enumerate(int_sites_preprocess):
    #             var = ''
    #             for index2, site in enumerate(sites):
    #                 if site in self.positions:
    #                     var += variants[index1][index2]
    #             out.append(var)
    #         # variants to integers
    #         x = from_variant_to_integer(out)
    #         return self.embed(x)

    def embed_seq(self, variants, verbose = False):
        out = from_variant_to_integer(variants)
        return self.embed(out)


    def embed(self, x):
        n, no_sites = x.size()
        vec_per_site = []
        for i in range(no_sites):
            # print (self.dict[int(x[0,i])])
            vec = torch.cat([self.dict[int(x[j, i])].view(1, -1) for j in range(n)])
            vec_per_site.append(vec)
        if not self.stacking:
            vec = torch.hstack(vec_per_site)
        else:
            if self.append > 0:
                vec = torch.stack(vec_per_site)[:, 0, :]
                # add append many dimensions filled with zeros to the beggining
                vec = torch.vstack((torch.zeros(size=(self.append,vec.size()[1])).double(), vec))
                #print (vec.size())
            else:
                vec = torch.stack(vec_per_site)[:,0,:]


        if self.projected == True:
            if self.mean is not None:
                vec = (vec - torch.tile(self.mean, (vec.size()[0], 1))) @ self.P
            else:
                vec = vec @ self.P
        else:
            if self.mean is not None:
                vec = vec - torch.tile(self.mean, (vec.size()[0], 1))

        if self.splits is not None:
            newy = vec.clone()
            for key in self.splits.keys():
                for b in self.splits[key]:
                    mask = vec[:, key] > b
                    ynew1 = torch.zeros(size=(vec.size()[0], 1)).double()
                    ynew1[mask] = vec[mask, key].view(-1, 1)
                    ynew2 = torch.zeros(size=(vec.size()[0], 1)).double()
                    ynew2[~mask] = vec[~mask, key].view(-1, 1)
                    newy[:, i] = ynew1.view(-1)
                    newy = torch.hstack((newy, ynew2))
            vec = newy
            vec = (vec - torch.tile(self.mean_split, (vec.size()[0], 1))) @ self.P_split
        return vec

    def load_projection(self, filename):
        self.P, self.mean = torch.load(filename)
        self.projected_components = self.P.size()[1]
        self.projected = True
        self.feature_names = ["proj_fea_" + str(i) for i in range(self.projected_components)]
        self.m = self.projected_components

    def save_projection(self, filename):
        torch.save([self.P, self.mean], filename)

    def project(self, n_sites, n_components=None):
        return self.project_joint_pca(n_sites, n_components=n_components)

    def get_m(self):
        return self.m

    def demean(self, n_sites, local=None):
        if local is None:
            xtest = torch.from_numpy(self.Opt.interval_number(dim=n_sites))
            embedding = self.embed(xtest)
            self.mean = torch.mean(embedding, dim=0)
        else:
            embedding = self.embed(local)
            self.mean = torch.mean(embedding, dim=0)

    def project_joint_pca(self, n_sites, n_components=None, demean=True):
        self.demean(n_sites)
        xtest = torch.from_numpy(self.Opt.interval_number(dim=n_sites))
        embedding = self.embed(xtest)

        # demean
        if demean:
            embedding = embedding - torch.tile(torch.mean(embedding, dim=0), (embedding.size()[0], 1))

        U, S, V = torch.pca_lowrank(A=embedding, q=25 * n_sites)
        self.projected_components = int((S >= self.tol).sum())
        if n_components is None:
            self.P = V[:, 0:self.projected_components] @ torch.diag(1. / (S[0:self.projected_components]))
        else:
            self.projected_components = n_components
            self.P = V[:, 0:self.projected_components] @ torch.diag(1. / (S[0:self.projected_components]))

        self.m = self.projected_components

    def dummy_projection(self, n_sites):
        self.demean(n_sites)
        xtest = torch.from_numpy(self.Opt.interval_number(dim=n_sites))
        embedding = self.embed(xtest)
        size = embedding.size()[1]
        self.P = torch.eye(size).double()
        self.projected_components = size
        self.m = size
        self.set_proto_names(n_sites)

    def set_proto_names(self, n_sites):
        self.proto_feature_names = ["VHSE_" + str(i) for i in range(8)] + ["Z_scales" + str(i) for i in
                                                                           range(5)] + [
                                       "Barley_0", "Barley_1"] \
                                   + ["PC_scores" + str(i) for i in range(11)]
        self.feature_names = [[fea + "_site_" + str(i) for fea in self.proto_feature_names] for i in range(n_sites)]
        self.feature_names = [item for sublist in self.feature_names for item in sublist]

    def project_individual_pca(self, n_sites, n_components=None, demean=True):
        self.demean(n_sites)
        xtest = torch.from_numpy(self.Opt.interval_number(dim=1))
        embedding = self.embed(xtest)

        if demean:
            embedding = embedding - torch.tile(torch.mean(embedding, dim=0), (embedding.size()[0], 1))

        U, S, V = torch.pca_lowrank(A=embedding, q=20)
        self.projected_components = int((S >= self.tol).sum())

        if n_components is None:
            P = V[:, 0:self.projected_components] @ torch.diag(1. / (S[0:self.projected_components]))
            self.P = torch.block_diag(*[P for _ in range(n_sites)])
        else:
            self.projected_components = n_components
            P = V[:, 0:self.projected_components] @ torch.diag(1. / (S[0:self.projected_components]))
            self.P = torch.block_diag(*[P for _ in range(n_sites)])

        self.m = self.projected_components * n_sites

    def set_projection(self):
        print("Using projected features with dim ", self.projected_components)
        self.projected = True

    def normalize_features(self, n_sites, local=None):
        if local is None:
            xtest = torch.from_numpy(self.Opt.interval_number(dim=n_sites))
            embedding = self.embed(xtest)
            norm = torch.max(torch.abs(embedding), dim=0)[0]
            self.P = self.P @ torch.diag(1. / norm)
        else:
            embedding = self.embed(local)
            norm = torch.max(torch.abs(embedding), dim=0)[0]
            self.P = torch.diag(1. / norm)

    def detach(self):
        self.P = self.P.detach()
        self.mean = self.mean.detach()


if __name__ == "__main__":
    dim = 5
    proxy = "-full"
    # proxy = ""
    data = '../../../data/amino-acid-features' + proxy + '.csv'
    # Emb = AminoAcidEmbedding(data = data)
    # Emb.project(dim)
    # Emb.set_projection()
    # Emb.normalize_features(dim)
    # Emb.save_projection("projection"+proxy+"-dim"+str(dim)+"-demean-norm.pt")
    #
    # Emb = AminoAcidEmbedding(data = data)
    # Emb.project(dim)
    # Emb.save_projection("projection"+proxy+"-dim"+str(dim)+"-demean.pt")

    Emb = AminoAcidEmbedding(data=data)
    Emb.dummy_projection(dim)
    Emb.set_projection()
    Emb.normalize_features(dim)
    Emb.save_projection("embedding" + proxy + "-dim" + str(dim) + "-demean-norm.pt")
