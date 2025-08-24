from mutedpy.protein_learning.embeddings.lookupmap import LookUpMap,LookUpMapMongo
from mutedpy.protein_learning.embeddings.lookupmap_static  import LookUpPickle, LookUpPickleDict
from stpy.embeddings.embedding import Embedding
import numpy as np
import torch
from typing import Union
from mutedpy.utils.protein_operator import ProteinOperator



class ProteinFeatureLoader(Embedding):

    def __init__(self,
                 data: Union[None,str],
                 aa_positions: list = [],
                 wild_type: Union[list,str] = [],
                 datatype: str = 'csv',
                 data_folder: str = '',
                 server: str = '',
                 credentials: str = '',
                 database: str = '',
                 project: str = '',
                 pandas_data = None,
                 header_mutation= 'Mutation',
                 embedding_name = 'values',
                 dict_callback = None,
                 lookupoptions = {},
                 ):

        self.op = ProteinOperator()
        self.datatype = datatype
        self.wild_type = wild_type
        self.aa_positions = aa_positions


        print ("DATATYPE",datatype)

        if datatype == 'csv':
            self.Embedding = LookUpMap(data=data_folder + data, header_mutation=header_mutation)
            self.feature_names = self.Embedding.feature_names

        elif datatype == 'pickle-torch':
            self.Embedding = LookUpPickle(data=data_folder + data, seqs=embedding_name,
                                          dict_callback = dict_callback,
                                          **lookupoptions)
            self.feature_names = self.Embedding.feature_names

        elif datatype == 'pickle-dict':
            self.Embedding = LookUpPickleDict(data=data_folder + data)
            #self.feature_names = self.Embedding.feature_names
        
        elif datatype == 'postgres':
            self.Embedding = LookUpPickleDict(server = server,
                                        credentials = credentials, database = database,
                                            project = project,  process_param_obj = self,
                                            embedding_name = embedding_name)

        elif datatype == 'postgres-dict':
            self.Embedding = LookUpPickleDict(data=data_folder + data)

        elif datatype == 'mongo':
            self.Embedding = LookUpMapMongo(data = None, server = server,
                                        credentials = credentials, database = database,
                                            project = project,  process_param_obj = self,
                                            embedding_name = embedding_name)
            self.feature_names = self.Embedding.feature_names

        elif datatype == 'mongo+preload':
            if data is not None:
                self.Embedding = LookUpMapMongo(data = data_folder + data, server = server,
                                                pandas_data = pandas_data,
                                        credentials = credentials, database = database,embedding_name=embedding_name,
                                                project = project, process_param_obj = self)
            else:
                self.Embedding = LookUpMapMongo(data=None, server=server, pandas_data=pandas_data,
                                                credentials=credentials, database=database,
                                                project=project,embedding_name=embedding_name,
                                                process_param_obj=self)
            self.feature_names = self.Embedding.feature_names

        self.m = self.Embedding.m



    def process_calable(self,x):
        return self.op.mutation(self.wild_type, self.aa_positions, self.op.translate_seq_to_variant(x))

    def embed_seq(self, x, verbose = True, mean = False, reshape = False):
        return self.Embedding.embed_seq(x, verbose=verbose, mean = mean, reshape = reshape)

    def embed(self, x):
        return self.Embedding.embed(x)

    def get_m(self):
        return self.Embedding.get_m()

    def close(self):
        if self.datatype[0:5] == 'mongo':
            self.Embedding.close()

    def connect(self):
        if self.datatype[0:5] == 'mongo':
            self.Embedding.connect()

    def get_db(self):
        return self.Embedding.db

    def __add__(self, x):
        obj = AddedProteinFeatureLoader(self, x)
        return obj

class AddedProteinFeatureLoader():

    def __init__(self, feature_loaders):
        self.feature_loaders =  feature_loaders

        self.m = self.get_m()
        self.feature_names = [fl.feature_names for fl in self.feature_loaders]
        self.feature_names = [food for sublist in self.feature_names for food in sublist]

    def get_m(self):
        return sum([fl.get_m() for fl in self.feature_loaders])

    def get_m_list(self):
        return [fl.get_m() for fl in self.feature_loaders]


    def connect(self):
        for fl in self.feature_loaders:
            #print (fl.datatype[0:5])
            if fl.datatype[0:5] == 'mongo':
                fl.connect()

    def close(self):
        for fl in self.feature_loaders:
            if fl.datatype[0:5] == 'mongo':
                fl.close()

    def embed(self, x):
        return torch.hstack([z.embed(x) for z in self.feature_loaders])

    def get_db(self):
       if self.feature_loaders[0].datatype[0:5] == 'mongo':
            return self.feature_loaders[0].Embedding.db

    def embed_seq(self, x,  verbose: bool = True):
        return torch.hstack([z.embed_seq(x, verbose = verbose) for z in self.feature_loaders])