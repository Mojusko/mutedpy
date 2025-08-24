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

from typing import Callable, Iterable, List, Optional, Sequence
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

# from your_module import Embedding, ProteinOperator

class LookUpMap(Embedding):
    """
    Core, file-backed feature lookup with a pluggable indexer.

    Typical workflow:
      - Construct from a CSV/HDF file where features have a prefix (default: 'fea').
      - If 'variant' column is missing, derive it from `header_mutation` using ProteinOperator.
      - Build a dictionary keyed by an integer index computed via `indexer(x_row)`.
      - Provide operations to restrict/transform features, update caches, and embed.

    Key customizations:
      - `feature_prefix`: select feature columns by prefix (default 'fea').
      - `indexer`: function mapping a 1D tensor row -> int (default: base-20 positional index).
      - `truncation`/`truncation_level`: optional binary truncation of outputs in `embed`.

    Notes:
      - The default indexer matches the typical "base-20" encoding used in your Mongo code.
      - For DB-backed subclasses, you can ignore the local dictionary or override methods as needed.
    """

    # ---------- Construction ----------
    def __init__(self,
                 data: str,
                 header_mutation: str = 'seq',
                 truncation: bool = False,
                 truncation_level: float = 10.,
                 feature_prefix: str = 'embedding',
                 indexer: Optional[Callable[[torch.Tensor], int]] = None):
        
        self.operator = ProteinOperator()
        self.truncation = truncation
        self.truncation_level = float(truncation_level)
        self.feature_prefix = feature_prefix
        self.indexer = indexer or (lambda row: self._baseN_index(row, base=20))

        # Load dataframe from supported formats
        df = self._load_dataframe(data)

        # Ensure 'variant'
        if 'variant' not in df.columns:
            df['variant'] = df[header_mutation].apply(self.operator.get_variant_code)

        # get sequences
        self.seqeunces = df['variant'].tolist()

        # Encode sequences
        x_np = self.operator.translate_mutation_series(df['variant'])  # shape: [N, d]
        self.x = torch.from_numpy(x_np).long()

        # Collect feature columns
        feature_cols = [c for c in df.columns if str(c).startswith(self.feature_prefix)]
        features_np = df[feature_cols].values.astype(float)
        self.phi = torch.from_numpy(features_np).double()

        # Bookkeeping
        self.feature_names = feature_cols
        self.N, self.d = self.x.size()
        self.m = self.phi.size(1)

        # Build the dictionary cache (int index -> feature row tensor)
        self.dictionary = {}
        self.int2var_dictionary ={}

        self._build_dictionary()

    # ---------- Loaders ----------
    def _load_dataframe(self, path: str) -> pd.DataFrame:
        ext = path.split(".")[-1].lower()
        if ext == "csv":
            return pd.read_csv(path)
        elif ext == "hdf":
            return pd.read_hdf(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Try csv or hdf.")

    # ---------- Indexing ----------

    def _get_indices(self, x: torch.Tensor) -> List[str]:
        """
        Compute integer indices for each row of x using `self.indexer`.
        """
        Xu = x.to(torch.uint8).contiguous().numpy()
        return [Xu[i].tobytes() for i in range(Xu.shape[0])]

    # ---------- Dictionary maintenance ----------
    def _build_dictionary(self):
        Xu = self.x.to(torch.uint8).contiguous().numpy()      
        for i in range(self.N):
            self.dictionary[self.seqeunces[i]] = self.phi[i]          
            self.int2var_dictionary[Xu[i].tobytes()] = self.seqeunces[i]
                   
    def _rebuild_dictionary(self):
        """
        Populate self.dictionary from self.x and self.phi using the current indexer.
        """
        self.dictionary.clear()
        for i in range(self.N):
            self.dictionary[self.seqeunces[i]] = self.phi[i]


    def _update_dictionary(self):
        """
        Rebuild dictionary after any change to self.phi (or self.x).
        """
        self.N, self.d = self.x.size()
        self.m = self.phi.size(1)
        self._rebuild_dictionary()

    # ---------- Feature selection / transforms ----------
    def restrict_by_std(self, std: float = 1.0):
        """
        Keep only features with std > `std`.
        """
        std_mask = torch.std(self.phi, dim=0) > float(std)
        self.feature_names = list(np.array(self.feature_names)[std_mask.cpu().numpy()])
        self.phi = self.phi[:, std_mask]
        old_m = self.m
        self.m = self.phi.size(1)
        self._update_dictionary()
        print(self.feature_names)
        print("New dimension:", self.m, "from", old_m)

    def restrict_by_name(self, names: Sequence[str] = ("surf",)):
        """
        Keep only features whose name contains any of the `names`.
        """
        old_m = len(self.feature_names)
        mask = np.array([any(n in fn for n in names) for fn in self.feature_names], dtype=bool)
        self.feature_names = list(np.array(self.feature_names)[mask])
        self.phi = self.phi[:, torch.from_numpy(mask)]
        self.m = self.phi.size(1)
        self._update_dictionary()
        print(self.feature_names)
        print("New dimension:", self.m, "from", old_m)

    def pca(self,
            std: float = 1.0,
            demean: bool = True,
            relative_var: bool = False,
            expl_var: float = 0.95,
            name: str = "volume"):
        """
        PCA via torch.pca_lowrank. Either keep components by absolute-std threshold
        or retain the smallest k s.t. cumulative explained variance >= expl_var.
        """
        print("Starting PCA calculation")

        Y = self.phi
        if demean:
            Y = Y - torch.mean(Y, dim=0, keepdim=True)

        q = int(min(Y.size(0), Y.size(1), 700))
        U, S, V = torch.pca_lowrank(Y, q=q)

        S_norm = S / torch.sum(S)
        if relative_var:
            k = int((torch.cumsum(S_norm, 0) < float(expl_var)).sum().item() + 1)
        else:
            k = int((S_norm > float(std)).sum().item())

        self.phi = Y @ V[:, :k]
        old_m = self.m
        self.m = self.phi.size(1)
        self._update_dictionary()
        self.feature_names = [f"{name}_proj_{i}" for i in range(self.m)]
        print(self.phi.size())
        print("New dimension:", self.m, "from", old_m)

    # ---------- Normalization/Standardization ----------
    def standardize(self, demean: bool = True):
        """
        Column-wise standardization (z-score normalization)
        """
        if demean:
            self.phi = self.phi - torch.mean(self.phi, dim=0, keepdim=True)

        # Use standard deviation instead of max for classical standardization
        std = torch.std(self.phi, dim=0, keepdim=True)
        # Avoid divide-by-zero; replace zeros with 1.0
        std = torch.where(std == 0, torch.ones_like(std), std)
        self.phi = self.phi / std
        self._update_dictionary()

    def normalize(self, demean: bool = False):
        """
        min-max normalization (outlier sensitive) to -1,1
        """
        if demean:
            self.phi = self.phi - torch.mean(self.phi, dim=0, keepdim=True)

        # Avoid divide-by-zero; replace zeros with 1.0
        min_vals = torch.min(self.phi, dim=0).values
        max_vals = torch.max(self.phi, dim=0).values
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)

        # Scale to [0,1] then shift to [-1,1]
        self.phi = (self.phi - min_vals) / range_vals
        self.phi = 2 * self.phi - 1

        self._update_dictionary()


    def l2_normalize(self, demean: bool = False):
        """
        L2 normalization of each row (optionally demean first).
        """
        if demean:
            self.phi = self.phi - torch.mean(self.phi, dim=0, keepdim=True)

        norms = torch.norm(self.phi, p=2, dim=1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        self.phi = self.phi / norms

        self._update_dictionary()
    
    def embed(self, x: torch.Tensor, cut: Optional[float] = None) -> torch.Tensor:
        """
        Return feature rows for x. If truncation is enabled, apply binary threshold.
        """
        n, _ = x.size()
        out = torch.zeros((n, self.m), dtype=torch.double)

        cut = self.truncation_level if cut is None else float(cut)

        keys = self.int2var_dictionary
        
        indices = self._get_indices(x)
        for j in range(n):

            if indices[j] not in keys:
                raise AssertionError(f"The index {indices[j]} was not found in the dictionary.")
            
            key = keys[indices[j]]

            if key in self.dictionary:
                row = self.dictionary[key]
                if self.truncation:
                    mask = row > cut
                    out[j, mask] = 0.0
                    out[j, ~mask] = 1.0
                else:
                    out[j, :] = row
            else:
                raise AssertionError(f"The feature {key} was not found.")
        return out

    def embed_seq(self, seq: List, verbose: bool = True,
                   mean: bool = False, reshape = False) -> torch.Tensor:
        """
        Embed sequences using the lookup dictionary.
        If `mean` is True, average the embeddings across all sequences.
        If `reshape` is True, reshape the output to (N, 1, m).
        """
        n = len(seq)
        out = torch.zeros((n, self.m), dtype=torch.double)
        if verbose:
            print(f"Embedding {n} sequences with dimension {self.m} features.")
        for i,s in enumerate(seq): 
            if s in self.dictionary:
                out[i, :] = self.dictionary[s]
            else:
                raise AssertionError(f"The sequence {s} was not found in the dictionary.")
        if mean:
            out = out.mean(dim=1, keepdim=True)
        return out 
    
    # ---------- (Optional) persistence ----------
    def save_dict(self, savefile: str):
        import pickle
        with open(savefile, "wb") as f:
            pickle.dump(self.dictionary, f)

    def load_dict(self, loadfile: str):
        import pickle
        with open(loadfile, "rb") as f:
            self.dictionary = pickle.load(f)
        # keep shapes in sync if possible
        if len(self.dictionary) and isinstance(next(iter(self.dictionary.values())), torch.Tensor):
            self.m = next(iter(self.dictionary.values())).numel()

class LookUpPickle(LookUpMap):

    def __init__(self, data : str = "", seqs: str = "", dict_callback = None, mean = False, reshape = False):
        self.seq_dictionary = {}
        self.mean = mean
        self.reshape = reshape
        emb = pickle.load(open(data,"rb"))
        seqs = pickle.load(open(seqs, "rb"))

        print ("Adding into keys...")
        print ("size: ", emb[0].size())
        print ("-------------------")

        for seq,e in zip(seqs,emb):

            if dict_callback is not None:

                if self.reshape:
                    self.seq_dictionary[dict_callback(seq)] = e.view(1,-1)
                else:
                    self.seq_dictionary[dict_callback(seq)] = e.unsqueeze(0)
            else:
                if self.reshape:
                    self.seq_dictionary[seq] = e.view(1, -1)
                else:
                    self.seq_dictionary[seq] = e.unsqueeze(0)

            self.m = e.size()[0]

        self.feature_names = ['lookup-pickle' + str(i) for i in range(self.m)]
        print ("LookupPickle created.")

class LookUpPickleDict(LookUpMap):

    def __init__(self, data : str):
        self.dictionary = pickle.load(open(data, "rb"))
        #print (self.dictionary)
        key, value = next(iter( self.dictionary.items()))
        #print (key, value)
        self.d = torch.Tensor(value).double().size()[1]
        self.N = 0
        self.mean = False
        self.reshape =True
        self.m = self.d

