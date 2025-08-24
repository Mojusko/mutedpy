from stpy.embeddings.embedding import Embedding
import pickle
from mutedpy.utils.protein_operator import ProteinOperator
import torch
from mutedpy.protein_learning.plm.zymctrl import ZymCTRL
from typing import Union

class ZymeCTRLEmbedding(Embedding):

    def __init__(self,
                 config = None,
                 device: str = 'cpu',
                 preloaded: Union [str,None] = None,
                 save_location: Union[str,None] = None,
                 mean:bool = False,
                 length:int = 91):
        self.config = config

        self.name = "zymctrl"
        self.mean = mean
        if mean:
            self.m = 1280
        else:
            self.m = 1280*length

        self.device = device
        self.feature_names = [self.name + "_" + str(i) for i in range(self.m)]
        self.save_location = save_location

        self.model = ZymCTRL(self.config)

        if preloaded is not None:
            if os.path.exists(preloaded):
                self.dict = pickle.load(open(preloaded, 'rb'))
            else:
                self.dict = {}
        else:
            self.dict = {}

    def embed_seq(self,x, device = None, batch_size = 10):
        if device is None:
            device = self.device

        out = []
        keys = self.dict.keys()
        for s in x:

            # if already embedded use the one
            if s in keys:
                out.append(self.dict[s])
            # embed with the pretrained model
            else:
                emb = self.model.embed([s])
                out.append(emb)
                self.dict[s] = emb

        # put together
        res = torch.cat(out).double()

        # take a mean if necessary
        if self.mean:
            res = torch.mean(res, dim = 1)
        return res

    def embed(self, x):
        """

        :param x: integer array of indicators
        :return: embeddings
        """
        P = ProteinOperator()
        # calculate sequences
        seqs = []
        for xx in x:
            s = P.translate_seq_to_variant(xx)
            seqs.append(s)
        return self.embed_seq(seqs)


if  __name__ == '__main__':
    from mutedpy.experiments.kemp.kemp_loader import *
    from omegaconf import OmegaConf

    x,y ,dts = load_first_round()
    seq_list = dts['variant'].values.tolist()[0:2]

    cfg = OmegaConf.load("/home/mojko/Documents/PhD_Projects/protein-design/zymectrl/config.yaml")

    # set the model to be used locally
    cfg.model.dir = "/home/mojko/Documents/PhD_Projects/protein-design/zymectrl/"

    # set the location of the tokenizer config
    cfg.model.tokenizer_config = "/home/mojko/Documents/PhD_Projects/protein-design/zymectrl/"

    embedding = ZymeCTRLEmbedding(config = cfg, mean = True)

    # embed a sequence
    emb = embedding.embed_seq(seq_list[0:2])

    print (emb.size())


