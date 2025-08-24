import esm.pretrained
from stpy.embeddings.embedding import Embedding
import pickle
from mutedpy.utils.protein_operator import ProteinOperator
import torch

class RandomEmbedding(Embedding):

    def __init__(self, m = 1280):
        self.m = m
        self.dict = {}
        self.feature_names = ["random_"+str(i) for i in range(self.m)]

    def get_m(self):
        return self.m

    def embed(self,x ):
        return self.embed_seq(x)

    def embed_seq(self, x):
        out = []
        keys = self.dict.keys()
        P = ProteinOperator()
        # calculate sequences
        for xx in x:
            s = P.translate_seq_to_variant(xx)
            # if already embedded use the one
            if s in keys:
                out.append(self.dict[s])
            # embed with the pretrained model
            else:
                z = torch.abs(torch.randn(size = (1,self.m)).double())*10
                self.dict[s] = z
                out.append(z)
        return torch.vstack(out).double()

