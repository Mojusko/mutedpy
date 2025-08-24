from stpy.embeddings.embedding import Embedding
import torch
import string
import numpy as np

class PretrainedEmbedding(Embedding):

    def __init__(self, model, params):

        d = torch.load(params)
        model.load_state_dict(d)
        model.eval()

        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        self.m = list(model.children())[-1].in_features
        self.one_hot_mapping = {}

        for i in range(20):
            aux = np.zeros(20)
            aux[i] = 1
            self.one_hot_mapping[i] = aux

    def get_m(self):
        return self.m

    def embed(self, x):
        xx = np.array([np.array([self.one_hot_mapping[int(s)] for s in sec]).flatten() for sec in x])
        xx = torch.FloatTensor(xx)
        return self.feature_extractor(xx).detach().double()
