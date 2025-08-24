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

from typing import Union, List, Optional, Tuple
import json
import pickle
import numpy as np
import pandas as pd
import torch
import psycopg2
import psycopg2.extras as pg_extras

from psycopg2 import sql
from mutedpy.protein_learning.embeddings.lookupmap_static import LookUpMap

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
        self.mean = False
        self.reshape = False

        if data is not None:
            file_format = data.split(".")[-1]

            if file_format == "csv":
                dts = pd.read_csv(data)
            elif file_format == "hdf":
                dts = pd.read_hdf(data)

            # this is the map to be preloaded
            dts['variant'] = dts[header_mutation].apply(self.operator.get_variant_code)
            self.x = torch.from_numpy(self.operator.translate_mutation_series(dts['variant'])).int()
            self.seq = dts['variant'].tolist()

        elif pandas_data is not None:
            dts = pandas_data
            self.x = torch.from_numpy(self.operator.translate_mutation_series(dts['variant'])).int()
            self.seq = dts['variant'].tolist()
            print ("Preloading from pandas data.", self.x.size(), self.x)

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

            for i in range(self.N):

                mutant = self.seq[i]
                if self.seq[i] not in self.dictionary:

                    if i%100==0:
                        print (i, "/", self.N, mutant)

                    res = self.db.find_one({"params": mutant})
                    if embedding_name not in res.keys():
                        raise AssertionError("Mutant"+mutant+" features not in DB.")

                    y = torch.from_numpy(np.array(res[embedding_name]))
                    self.m = y.size()[0]
                    self.dictionary[self.seq[i]] = y

            print ("Features loaded from MongoDB.")
            #self.feature_names = dts.columns[1:-2]
        self.client.close()

        del self.client
        del self.db


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

        out = []
        keys = self.dictionary.keys()

        for j in range(n):
            mutant = x[j]

            if mutant in keys:
               res = self.dictionary[mutant]

            else:
                if verbose:
                    print(mutant + " Not found in dictionary. Retrieving from MongoDB", j, '/', n)

                res = self.db.find_one({"params": mutant},{self.embedding_name:1})[self.embedding_name]
                self.dictionary[mutant] = res

            if res is None:
                raise AssertionError("Mutant"+mutant+"not in DB.")
            else:
                # calculate mean of the embedding around dim = 1
                if mean or self.mean:
                    out += [torch.mean(torch.from_numpy(np.array(res)), dim = 1)]
                else:
                    if reshape or self.reshape:
                        out += [torch.from_numpy(np.array(res)).view(-1)]
                    else:
                        out += [torch.from_numpy(np.array(res))]
                del res

        # stacks together
        out = torch.stack(out)

        return out




class LookUpMapPostgres(LookUpMap):
    """
    PostgreSQL-backed lookup map.

    Param semantics:
      - server:        "user@host" or "user@host:port"
      - credentials:   password (string, not a file)
      - database:      database name
      - project:       table name (equivalent to a Mongo collection)
      - embedding_name: column name with the embedding (default "values")

    Schema expectation (minimal):
      CREATE TABLE your_table (
        sequence TEXT PRIMARY KEY,
        values   JSONB   -- or DOUBLE PRECISION[] / DOUBLE PRECISION[][]
      );

      -- Optional feature names table:
      CREATE TABLE feature_names (
        project TEXT PRIMARY KEY,
        names   JSONB
      );
    """

    def __init__(self,
                 server: str,
                 credentials: str,     # password
                 database: str,
                 project: str,         # table name
                 data: Optional[str],
                 header_mutation: str = 'mutation',
                 process_param_obj=None,
                 pandas_data: Optional[pd.DataFrame] = None,
                 embedding_name: str = "values"):

        self.process_param_obj = process_param_obj
        self.operator = ProteinOperator()
        self.embedding_name = embedding_name
        self.mean = False
        self.reshape = False

        self.x = None
        self.seq = None
        self.dictionary = {}
        self.seq_dictionary = {}

        # ---------- Parse server "user@host[:port]" + password ----------
        user, host, port = self._parse_user_host_port(server)
        password = credentials  # per your spec
        self.dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        self.database = database
        self.project = project               # table name
        self.table_ident = sql.Identifier(self.project)
        self.col_embedding_ident = sql.Identifier(self.embedding_name)

        # ---------- Preload from CSV/HDF/PKL or pandas ----------
        _skip_warm_db_preload = False

        if data is not None:
            file_format = data.split(".")[-1].lower()

            if file_format in ("csv", "hdf"):
                dts = pd.read_csv(data) if file_format == "csv" else pd.read_hdf(data)
                dts['variant'] = dts[header_mutation].apply(self.operator.get_variant_code)
                self.x = torch.from_numpy(self.operator.translate_mutation_series(dts['variant'])).int()
                self.seq = dts['variant'].tolist()

            elif file_format in ("pkl", "pickle"):
                with open(data, "rb") as f:
                    loaded = pickle.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError("Pickle must contain a dict mapping keys -> embeddings.")
                # normalize to tensors
                norm = {}
                for k, v in loaded.items():
                    t = v if isinstance(v, torch.Tensor) else torch.from_numpy(np.array(v))
                    norm[k] = t
                self.dictionary = norm
                if not self.dictionary:
                    raise ValueError("Pickle dictionary is empty; cannot infer embedding size.")
                sample = next(iter(self.dictionary.values()))
                try:
                    self.d = sample.size()[1]
                except Exception:
                    self.d = sample.size()[0]
                self.N = 0
                self.m = self.d
                _skip_warm_db_preload = True
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

        elif pandas_data is not None:
            dts = pandas_data
            self.x = torch.from_numpy(self.operator.translate_mutation_series(dts['variant'])).int()
            self.seq = dts['variant'].tolist()
            print("Preloading from pandas data.", self.x.size(), self.x)

        # ---------- Connect once to infer dims / load names / warm-preload ----------
        self.connect(verbose=False)

        if not hasattr(self, "d"):
            try:
                self.N, self.d = self.x.size()
            except Exception:
                row = self._peek_one_embedding()
                if row is None:
                    raise RuntimeError(
                        f"No embeddings found in table '{self.project}'. Cannot infer dimensions."
                    )
                res_t = torch.from_numpy(np.array(row))
                try:
                    self.d = res_t.size()[1]
                except Exception:
                    self.d = res_t.size()[0]
                self.N = 0
            self.m = self.d

        # Feature names
        self.feature_names = self._load_feature_names() or [f'lookup{i}' for i in range(self.d)]

        # Warm-preload selected mutants, unless a pickle already provided the cache
        if (self.seq is not None) and (self.N > 0) and (not _skip_warm_db_preload):
            for i in range(self.N):
                mutant = self.seq[i]
                if mutant not in self.dictionary:
                    if i % 100 == 0:
                        print(i, "/", self.N, mutant)
                    y = self._fetch_embedding_by_sequence(mutant)
                    if y is None:
                        raise AssertionError(f"Mutant {mutant} features not in DB.")
                    y_t = torch.from_numpy(np.array(y))
                    self.m = y_t.size()[0]
                    self.dictionary[mutant] = y_t
            print("Features loaded from PostgreSQL.")

        self.close(verbose=False)

    # ---------- Connection management ----------
    def connect(self, verbose: bool = False):
        if verbose:
            print("Connecting to PostgreSQL.")
        self.conn = psycopg2.connect(self.dsn)
        pg_extras.register_default_jsonb(self.conn, loads=json.loads)
        self.conn.autocommit = True

    def close(self, verbose: bool = False):
        if verbose:
            print("Closing PostgreSQL connection.")
        try:
            self.conn.close()
        finally:
            del self.conn

    # ---------- Helpers ----------
    def _parse_user_host_port(self, server: str) -> Tuple[str, str, int]:
        """
        Parse 'user@host' or 'user@host:port' -> (user, host, port).
        """
        if "@" not in server:
            raise ValueError("server must be 'user@host' or 'user@host:port'.")
        user, hostpart = server.split("@", 1)
        if not user or not hostpart:
            raise ValueError("Invalid 'server' format; missing user or host.")
        if ":" in hostpart:
            host, port_str = hostpart.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                raise ValueError("Port must be an integer.")
        else:
            host, port = hostpart, 5432
        return user, host, port

    def _peek_one_embedding(self):
        with self.conn.cursor(cursor_factory=pg_extras.RealDictCursor) as cur:
            q = sql.SQL("SELECT {col} FROM {tbl} LIMIT 1").format(
                col=self.col_embedding_ident, tbl=self.table_ident
            )
            cur.execute(q)
            row = cur.fetchone()
            if not row:
                return None
            return row[self.embedding_name]

    def _fetch_embedding_by_sequence(self, mutant: str):
        with self.conn.cursor(cursor_factory=pg_extras.RealDictCursor) as cur:
            q = sql.SQL("SELECT {col} FROM {tbl} WHERE sequence = %s LIMIT 1").format(
                col=self.col_embedding_ident, tbl=self.table_ident
            )
            cur.execute(q, (mutant,))
            row = cur.fetchone()
            if not row:
                return None
            return row[self.embedding_name]
        

    def _fetch_embeddings_batch(self, mutants: list[str], batch: int = 1000):
        if not mutants:
            return []
        out = []
        q = sql.SQL("""
            SELECT u.ord, t.{col}
            FROM unnest(%s::text[]) WITH ORDINALITY AS u(seq, ord)
            LEFT JOIN {tbl} t ON t.sequence = u.seq
            ORDER BY u.ord
        """).format(col=self.col_embedding_ident, tbl=self.table_ident)

        with self.conn.cursor() as cur:  # default cursor is faster than RealDictCursor
            for i in range(0, len(mutants), batch):
                print(f"Fetching batch {i // batch + 1} of {len(mutants) // batch + 1}...")
                sub = mutants[i:i+batch]
                cur.execute(q, (sub,))
                out.extend(row[1] for row in cur.fetchall())  # emb aligned to sub
        return out

    def _load_feature_names(self) -> Optional[List[str]]:
        try:
            with self.conn.cursor(cursor_factory=pg_extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT names FROM feature_names WHERE project = %s LIMIT 1",
                    (self.project,),
                )
                r = cur.fetchone()
                if r and r.get("names"):
                    return r["names"]
        except Exception:
            return None
        return None

    def embed_seq(self, x: List,
                  verbose: bool = True,
                  mean: bool = True,
                  reshape: bool = False) -> torch.Tensor:
        n = len(x)
        if mean:
            out = torch.zeros(size = (n, self.m), dtype=torch.float64)
        else:
            raise NotImplementedError("mean=False is not implemented in embed_seq()")
        keys = self.dictionary.keys()

        # first check if they are in the local dictionary
        to_fetch = []
        positions=[]
        for j in range(n):
            mutant = x[j]
            if mutant in keys:
                res = self.dictionary[mutant]
                out[j, :] = res if isinstance(res, torch.Tensor) else torch.from_numpy(np.array(res))
            else:
                to_fetch.append(mutant)
                positions.append(j)

        if len(to_fetch)>0: 
            res = self._fetch_embeddings_batch(to_fetch)
            
            for j, fetch in enumerate(to_fetch):
                self.dictionary[fetch] = res[j] if isinstance(res[j], torch.Tensor) else torch.from_numpy(np.array(res[j]))
                out[positions[j],:] = res[j] if isinstance(res[j], torch.Tensor) else torch.from_numpy(np.array(res[j]))
        return out

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        n, d = x.size()
        out = torch.zeros(n, self.m).double()

        indices = self.get_index(x)
        keys = self.dictionary.keys()

        for j in range(n):
            idx = int(indices[j])
            if idx in keys:
                cached = self.dictionary[idx]
                out[j, :] = cached if isinstance(cached, torch.Tensor) else torch.from_numpy(np.array(cached))
            else:
                if self.process_param_obj is None or not hasattr(self.process_param_obj, 'process_calable'):
                    raise RuntimeError("process_param_obj with .process_calable(tensor) is required for embed().")
                mutant = self.process_param_obj.process_calable(x[j, :])  # expected to return the sequence string
                res = self._fetch_embedding_by_sequence(mutant)
                if res is None:
                    raise AssertionError(f"Mutant {mutant} not in DB.")
                emb_t = res if isinstance(res, torch.Tensor) else torch.from_numpy(np.array(res))
                out[j, :] = emb_t
                self.dictionary[idx] = emb_t

        return out

    def get_all_sequences_in_db(self):
        """
        Retrieve all sequences stored in the PostgreSQL database.
        Returns:
            List[str]: A list of all sequences.
        """
        with self.conn.cursor(cursor_factory=pg_extras.RealDictCursor) as cur:
            cur.execute(sql.SQL("SELECT sequence FROM {}").format(self.table_ident))
            rows = cur.fetchall()
            return [row['sequence'] for row in rows]
