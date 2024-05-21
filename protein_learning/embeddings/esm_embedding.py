import esm.pretrained
from stpy.embeddings.embedding import Embedding
import pickle
from mutedpy.utils.protein_operator import ProteinOperator
import torch
import os

class ESMEmbedding(Embedding):

    def __init__(self, name = 'esm-1v', device = 'cpu', preloaded = None, save_location = None, mean = False, length = 91):
        if name == 'esm-1v':
            self.model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_5()
            self.batch_converter = alphabet.get_batch_converter()
        elif name == 'esm-2':
            self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.batch_converter = alphabet.get_batch_converter()
        else:
            raise AssertionError("Not Implemented.")

        self.model.eval()
        self.mean = mean
        if mean:
            self.m = 1280
        else:
            self.m = 1280*length
        self.device = device
        self.feature_names = [name + "_" + str(i) for i in range(self.m)]

        if preloaded is not None:
            if os.path.exists(preloaded):
                self.dict = pickle.load(open(preloaded, 'rb'))
            else:
                self.dict = {}
        else:
            self.dict = {}


    def dump_embeddings(self, name):
        pickle.dump(self.dict, open(name,"wb"))

    def load_embeddings(self,name):
        self.dict = pickle.load(open(name,"rb"))

    def get_m(self):
        return self.m

    def embed_seq_all_new(self, x):
        batch_labels, batch_strs, batch_tokens = self.batch_converter([("dummy", s) for s in x])
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            if self.mean:
                z = torch.mean(token_representations, dim = 1)
            else:
                z = token_representations
        return z
    def embed_seq(self,x, device = None, batch_size = 10, verbose = False):
        if device is None:
            device = self.device
        out = []
        keys = self.dict.keys()
        self.model.to(device)
        for s in x:
            # if already embedded use the one
            if s in keys:
                out.append(self.dict[s])
            # embed with the pretrained model
            else:
                batch_labels, batch_strs, batch_tokens = self.batch_converter([("dummy", s)])
                with torch.no_grad():
                    results = self.model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
                    token_representations = results["representations"][33]
                    if self.mean:
                        z = token_representations[0, :, :].mean(0)
                    else:
                        z = token_representations[0, :, :]
                self.dict[s] = z
                out.append(z)
        return torch.stack(out).double().to('cpu')

    def embed(self, x):
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
                batch_labels, batch_strs, batch_tokens = self.batch_converter([("dummy",s)])
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                    token_representations = results["representations"][33]
                    if self.mean:
                        z = token_representations[0,:,:].mean(0)
                    else:
                        z = token_representations[0, :, :]
                self.dict[s] = z
                out.append(z)
        return torch.stack(out).double()

if  __name__ == '__main__':
    import time
    from mutedpy.experiments.kemp.kemp_loader import *
    x,y ,dts = load_first_round()


    seq_list = dts['variant'].values.tolist()[0:100]


    embedding = ESMEmbedding('esm-1v')

    t0 = time.time()
    Phi = embedding.embed_seq_all_new(seq_list)
    t1 = time.time()
    print (t1-t0)
    print (Phi.size())

    t0 = time.time()
    Phi = embedding.embed_seq(seq_list)
    t1 = time.time()
    print(t1 - t0)
    print (Phi.size())

