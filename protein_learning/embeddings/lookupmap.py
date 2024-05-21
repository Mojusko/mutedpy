import pymongo
from stpy.embeddings.embedding import Embedding
import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from typing import Callable, Type, Union, Tuple, List
import torch
from stpy.test_functions.protein_benchmark import ProteinOperator
from stpy.test_functions.protein_benchmark import ProteinBenchmark
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

class LookUpMap(Embedding):

    def __init__(self, data:  str,
                 header_mutation: str = 'mutation',
                 truncation: bool = True,
                 truncation_level: float = 10.):

        operator = ProteinOperator()

        file_format = data.split(".")[-1]

        if file_format == "csv":
            dts = pd.read_csv(data)

        elif file_format == "hdf":
            dts = pd.read_hdf(data)


        if 'variant' not in dts.columns:
            dts['variant'] = dts[header_mutation].apply(operator.get_variant_code)

        self.x = torch.from_numpy(operator.translate_mutation_series(dts['variant']))
        column_names = []

        for col in dts.columns:
            if col[0:3] == 'fea':
                column_names.append(col)

        features = dts[column_names]
        self.y = torch.from_numpy(features.values.astype(float))

        self.feature_names = column_names
        self.truncation_level = truncation_level
        self.truncation = truncation
        self.m = self.y.size()[1]
        self.N = self.y.size()[0]
        self.d = self.x.size()[1]
        self.dictionary = {}
        self.f = lambda x: torch.sum(torch.tensor([int(x[i])  for i in range(self.d)],dtype = torch.long))
        #self.f = lambda x: torch.Tensor([torch.sum(x[i]*i) for i in range(self.d)]))]).int()
        indices = self.get_index(self.x)

        for i in range(self.N):
            self.dictionary[int(indices[i])] = self.y[i]

    def restrict_to_varaint(self, std=1.):

        mask = torch.std(self.y, dim=0) > std
        self.feature_names = self.feature_names[mask.view(-1)]
        self.y = self.y[:, mask]
        old_m = self.m
        self.m = self.y.size()[1]
        indices = self.get_index(self.x)
        for i in range(self.N):
            self.dictionary[int(indices[i])] = self.y[i]

        print(self.feature_names)
        print("New dimension:", self.m, "from", old_m)

    def restrict_by_name(self, names=["surf"]):
        old_m = len(self.feature_names)
        mask = torch.zeros(size=(old_m, 1)).view(-1).bool()
        for name in names:
            for i in range(old_m):
                if name in self.feature_names[i]:
                    mask[i] = True

        # update features
        self.feature_names = self.feature_names[mask.view(-1)]
        self.y = self.y[:, mask]
        self.m = self.y.size()[1]
        indices = self.get_index(self.x)
        for i in range(self.N):
            self.dictionary[int(indices[i])] = self.y[i]
        print(self.feature_names)
        print("New dimension:", self.m, "from", old_m)

    def pca(self, std=1., demean=True, relative_var=False, expl_var=0.95, name="volume"):
        print("Starting pca calculation")

        if demean:
            self.y = self.y - torch.tile(torch.mean(self.y, dim=0), (self.y.size()[0], 1))

        (U, S, V) = torch.pca_lowrank(self.y, q=np.min([self.y.size()[0], self.y.size()[1], 700]))
        if relative_var:
            S = S / torch.sum(S)
            k = int(torch.sum(torch.cumsum(S, 0) < expl_var) + 1)
        else:
            S = S / torch.sum(S)
            k = int(torch.sum(S > std))

        self.y = torch.matmul(self.y, V[:, :k])

        print(self.y.size())
        old_m = self.m
        self.m = self.y.size()[1]

        self.update()
        print("New dimension:", self.m, "from", old_m)
        self.feature_names = [name + "_proj_" + str(i) for i in range(self.m)]

    def split_features(self, xtest=None, bandwidth=0.2, demean=True, normalize=True):
        n, no_features = self.y.size()
        newy = self.y.clone()
        for i in range(no_features):
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.y[:, i].numpy().reshape(-1, 1))
            xmin = torch.min(self.y[:, i])
            xmax = torch.max(self.y[:, i])
            x_plot = torch.linspace(xmin, xmax, 1024)
            log_dens = kde.score_samples(x_plot.view(-1, 1))
            breakpoints = argrelextrema(log_dens, np.less)

            if len(breakpoints[0]) > 0:
                for index, b in enumerate(breakpoints[0]):
                    print("breakpoint detected in feature:", i, "index:", index)
                    mask = self.y[:, i] > float(x_plot[b])
                    ynew1 = torch.zeros(size=(self.y.size()[0], 1)).double()
                    ynew1[mask] = self.y[mask, i].view(-1, 1)
                    ynew2 = torch.zeros(size=(self.y.size()[0], 1)).double()
                    ynew2[~mask] = self.y[~mask, i].view(-1, 1)
                    newy[:, i] = ynew1.view(-1)
                    newy = torch.hstack((newy, ynew2))
                    feature_name = self.feature_names[i]
                    self.feature_names[i] = feature_name + "_splitted"
                    self.feature_names = np.append(self.feature_names, feature_name + "_splitted_" + str(index))
            else:
                print("no breakpoint", i)
        self.y = newy
        self.m = self.y.size()[1]

        if demean:
            self.y = self.y - torch.tile(torch.mean(self.y, dim=0), (self.y.size()[0], 1))
            self.update()

        if normalize:
            self.normalize(xtest=xtest)

    def update(self):
        indices = self.get_index(self.x)
        for i in range(self.N):
            self.dictionary[int(indices[i])] = self.y[i]

    def normalize(self, xtest=None, demean=True):
        if xtest is None:
            xtest = torch.from_numpy(self.Opt.interval_number(dim=n_sites))
        else:
            xtest = self.x

        if demean:
            self.y = self.y - torch.tile(torch.mean(self.y, dim=0), (self.y.size()[0], 1))

        norm = torch.max(torch.abs(self.y), dim=0)[0]
        self.y = self.y @ torch.diag(1. / norm)
        self.update()

    def get_index(self, x):
        return torch.stack([self.f(x_i) for i, x_i in enumerate(torch.unbind(x, dim=0), 0)], dim=0)

    def mask_if_inside(self, x: torch.Tensor) -> torch.Tensor:
        n, d = x.size()
        out = torch.zeros((n, 1), dtype=bool).view(-1)
        where = torch.zeros((n, 1), dtype=int).view(-1)
        indices = self.get_index(x)
        keys = self.dictionary.keys()

        for j in range(n):
            if int(indices[j]) in keys:
                out[j] = True
                where[j] = int(indices[j])
        return out, where

    def embed(self, x: torch.Tensor, cut=None) -> torch.Tensor:
        n, d = x.size()
        out = torch.zeros(n, self.m).double()

        if cut is None:
            cut = self.truncation_level

        indices = self.get_index(x)
        keys = self.dictionary.keys()

        for j in range(n):
            if int(indices[j]) in keys:

                if self.truncation:
                    mask = self.dictionary[int(indices[j])] > cut
                    out[j, mask] = 0.
                    out[j, ~mask] = 1.
                else:
                    out[j, :] = self.dictionary[int(indices[j])]

            else:
                raise AssertionError("The feature", int(indices[j])," was not found.")
        return out


class LookUpMapMongo(LookUpMap):


    def __init__(self,
                 server: str,
                 credentials: str,
                 database: str,
                 project: str,
                 data: Union[str,None],
                 header_mutation: str = 'mutation',
                 process_param_obj = None, # a lambda function that proposed how to relate an index in the db
                 pandas_data = None,
                 embedding_name = "values"
                 ):

        self.process_param_obj = process_param_obj
        self.operator = ProteinOperator()
        self.embedding_name = embedding_name

        if data is not None:
            file_format = data.split(".")[-1]

            if file_format == "csv":
                dts = pd.read_csv(data)
            elif file_format == "hdf":
                dts = pd.read_hdf(data)

            # this is the map to be preloaded
            dts['variant'] = dts[header_mutation].apply(self.operator.get_variant_code)
            self.x = torch.from_numpy(self.operator.translate_mutation_series(dts['variant']))

        elif pandas_data is not None:
            dts = pandas_data
            self.x = torch.from_numpy(self.operator.translate_mutation_series(dts['variant']))

        # initialize the database connection
        with open(credentials, 'r') as f:
            credentials = f.read()

        # load database
        self.server = credentials + "@" + server
        self.client = pymongo.MongoClient("mongodb://" + self.server)
        self.database = database
        self.project = project
        self.db = self.client[self.database][self.project]

        # these are the features
        try:
            self.N, self.d = self.x.size()
        except:
            res = torch.from_numpy(np.array(self.db.find_one({})[embedding_name]))
            print (res)
            try:
                self.d = res.size()[1]
            except:
                self.d = res.size()[0]
            self.N = 0

        self.m = self.d
        self.dictionary = {}
        self.seq_dictionary = {}
        # load the feature names
        db_names = self.client["feature-names"][self.project]
        res =list(db_names.find({}))
        if len(res)>0:
            self.feature_names = res[0]['names']
        else:
            self.feature_names = ['lookup'+str(i) for i in range(self.d)]


        if data is not None or pandas_data is not None:
            indices = self.get_index(self.x)

            for i in range(self.N):
                mutant = self.process_param_obj.process_calable(self.x[i, :])
                if i%1000==0:
                    print (i, "/", self.N, mutant)
                res = self.db.find_one({"params": mutant})
                y = torch.from_numpy(np.array(res[embedding_name]))
                self.m = y.size()[0]
                self.dictionary[int(indices[i])] = y

            print ("Features loaded from MongoDB.")
            #self.feature_names = dts.columns[1:-2]
        self.client.close()

        del self.client
        del self.db


    def f(self,x):
        return torch.sum(torch.from_numpy(np.array([int(x[i]) * (20 ** i) for i in range(self.d)])))


    def connect(self, verbose = False):
        if verbose:
            print ("Connecting to MongoDB.")
        self.client = pymongo.MongoClient("mongodb://" + self.server)
        self.db = self.client[self.database][self.project]

    def close(self, verbose = False):
        if verbose:
            print ("Closing connection to MongoDB.")
        self.client.close()
        del self.client
        del self.db

    def embed_seq(self, x: List,
                  verbose: bool = True,
                  mean: bool = False,
                  reshape: bool = False
                  )-> torch.Tensor:
        n = len(x)

        #out = torch.zeros(n, self.m).double()
        out = []
        keys = self.seq_dictionary.keys()

        for j in range(n):
            mutant = x[j]

            if mutant in keys:
               res = self.seq_dictionary[mutant]
            else:
                if verbose:
                    print("Retrieving from MongoDB", j, '/', n)

                res = self.db.find_one({"params": mutant},{self.embedding_name:1})[self.embedding_name]
                self.seq_dictionary[mutant] = res
            if res is None:
                raise AssertionError("Mutant"+mutant+"not in DB.")

            else:
                if mean or self.mean:
                    # print (torch.from_numpy(np.array(res)).size())
                    out += [torch.mean(torch.from_numpy(np.array(res)), dim = 1)]
                else:
                    if reshape or self.reshape:
                        out += [torch.from_numpy(np.array(res)).view(-1)]
                    else:
                        out += [torch.from_numpy(np.array(res))]
                del res

        out = torch.cat(out)
        #print (out.size())
        return out

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        n, d = x.size()
        out = torch.zeros(n, self.m).double()

        indices = self.get_index(x)
        keys = self.dictionary.keys()

        for j in range(n):
            if int(indices[j]) in keys:
                out[j, :] = self.dictionary[int(indices[j])]
            else:
                # get it from mongodb
                print ("Retrieving from MongoDB",j,'/',n)
                mutant = self.process_param_obj.process_calable(x[j,:])
                res = self.db.find_one({"params":mutant})
                out[j,:] = torch.from_numpy(np.array(res[self.embedding_name]))
        return out

class LookUpPickle(LookUpMapMongo):

    def __init__(self, data : str = "", seqs: str = "", dict_callback = None, mean = False, reshape = False):
        self.seq_dictionary = {}
        self.mean = mean
        self.reshape = reshape
        emb = pickle.load(open(data,"rb"))
        seqs = pickle.load(open(seqs, "rb"))
        print ("Adding into keys...")
        for seq,e in zip(seqs,emb):

            if dict_callback is not None:

                if self.reshape:
                    self.seq_dictionary[dict_callback(seq)] = e.view(1,-1)
                else:
                    self.seq_dictionary[dict_callback(seq)] = e
            else:
                if self.reshape:
                    self.seq_dictionary[seq] = e.view(1, -1)
                else:
                    self.seq_dictionary[seq] = e

            self.m = e.size()[0]

        self.feature_names = ['lookup-pickle' + str(i) for i in range(self.m)]
        print ("LookupPickle created.")
