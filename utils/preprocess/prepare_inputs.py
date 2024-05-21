import os
import binascii
import scipy.spatial
import biotite.structure as st
from biotite.structure.io.pdb import PDBFile
from tqdm import tqdm
import lmdb
import io
import gzip
import json
import pickle as pkl
from typing import Callable, List
import argparse
import torch
from torch_geometric.data import Data
import traceback
import mutedpy.protein_learning.neural_network_architectures.atom3d.protein.sequence as seq
import mutedpy.protein_learning.neural_network_architectures.atom3d.util.graph as gr
import mutedpy.protein_learning.neural_network_architectures.atom3d.util.formats as fo

import multiprocessing as mp

import numpy as np
import pandas as pd
from mutedpy.utils.loaders.loader_basel import BaselLoader
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations, order_mutations, create_neural_mutations



def str2bool(v: str) -> bool:
	"""Converts str to bool.
	Parameters
	----------
	v: str,
		String element
	Returns
	-------
	boolean version of v
	"""
	v = v.lower()
	if v == "true":
		return True
	elif v == "false":
		return False
	else:
		raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


def serialize(x, serialization_format):
	"""
	Serializes dataset `x` in format given by `serialization_format` (pkl, json, msgpack).
	"""
	if serialization_format == 'pkl':
		# Pickle
		# Memory efficient but brittle across languages/python versions.
		return pkl.dumps(x)
	elif serialization_format == 'json':
		# JSON
		# Takes more memory, but widely supported.
		serialized = json.dumps(
			x, default=lambda df: json.loads(
				df.to_json(orient='split', double_precision=6))).encode()
	else:
		raise RuntimeError('Invalid serialization format')
	return serialized


def make_lmdb_dataset(protein_list: List[str],
					  output_lmdb: str,
					  filter_fn: Callable = None,
					  serialization_format: str = 'json'):
	num_examples = len(protein_list)

	env = lmdb.open(str(output_lmdb), map_size=int(1e13))

	with env.begin(write=True) as txn:
		try:
			id_to_idx = {}
			i = 0

			for x in tqdm.tqdm(protein_list, total=num_examples):
				if filter_fn is not None and filter_fn(x):
					continue
				# Add an entry that stores the original types of all entries
				x['types'] = {key: str(type(val)) for key, val in x.items()}
				# ... including itself
				x['types']['types'] = str(type(x['types']))
				buf = io.BytesIO()
				with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
					f.write(serialize(x, serialization_format))
				compressed = buf.getvalue()
				result = txn.put(str(i).encode(), compressed, overwrite=False)
				if not result:
					raise RuntimeError(f'LMDB entry {i} in {str(output_lmdb)} '
									   'already exists')
				id_to_idx[x['id']] = i
				i += 1
		finally:
			txn.put(b'num_examples', str(i).encode())
			txn.put(b'serialization_format', serialization_format.encode())
			txn.put(b'id_to_idx', serialize(id_to_idx, serialization_format))


def prepare_graphs(mutant_list, base_dir, output_dir,  make_lmdb=False):

	input_dir = f"{base_dir}"
	failures = []

	if make_lmdb:
		protein_list = []



	for pdb_id, fitness in zip(mutant_list['Mutation'],mutant_list['LogFitness']):

		try:
			pdb_file = f"{input_dir}/{pdb_id}.pdb"

			# load bitotite
			file = PDBFile.read(pdb_file)
			structure = file.get_structure(include_bonds=True, model=1)
			# reduce to one chain
			mask = structure.chain_id == "B"
			structure = structure[mask]

			for i in range(112,124,1):
				mask = np.logical_or(structure.res_id == i, mask)
			structure = structure[mask]

			# save to temp-file
			receptor_file = PDBFile()
			receptor_file.set_structure(structure)
			name = binascii.b2a_hex(os.urandom(15))
			receptor_file.write("/tmp/"+name+".pdb")
			# load again
			pdb_file = "/tmp/"+name+".pdb"

			protein_dict = {}
			struct = fo.read_any(pdb_file)
			atoms = fo.bp_to_df(struct)
			sequence = seq.get_chain_sequences(atoms)

			protein_dict["atoms"] = atoms
			protein_dict["seq"] = sequence
			protein_dict['fitness'] = fitness

			node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(protein_dict["atoms"],  edge_dist_cutoff=4.5)
			prot_graph = Data(node_feats, edge_index, edge_feats, y=fitness, pos=pos)
			torch.save(prot_graph, f"{output_dir}/{pdb_id}.pth")

			print(f"{pdb_id} processed", flush=True)

			if make_lmdb:
				protein_dict['pdb_id'] = pdb_id
				protein_list.append(protein_dict)

		except Exception as e:
			print(f"Could not process {pdb_id} because of {e}", flush=True)
			traceback.print_exc()
			failures.append(pdb_id)
			continue




	if make_lmdb:
		make_lmdb_dataset(protein_list, output_lmdb=f"{output_dir}")

	if len(failures):
		with open(f"{output_dir}/failures.txt", "w") as f:
			for failure in failures:
				f.write(f"{failure}\n")

def prepare_graphs_parallel(mutant_list, base_dir, output_dir,  make_lmdb=False):
	#def prepare(id, fitness):
	#	prepare_graph(base_dir, output_dir, fitness, id)
	cores = mp.cpu_count()
	pool = mp.Pool(cores)
	results = pool.map(calculate, [(base_dir, output_dir, pdb_id, fitness) for (pdb_id, fitness) in zip(mutant_list['Mutation'],mutant_list['LogFitness'])])

def calculate(args):
	prepare_graph(args[0],args[1],args[3],args[2])

def prepare_graph(input_dir,output_dir,fitness,pdb_id):
	try:
		pdb_file = f"{input_dir}/{pdb_id}.pdb"

		# load bitotite
		file = PDBFile.read(pdb_file)
		structure = file.get_structure(include_bonds=True, model=1)
		# reduce to one chain
		mask = structure.chain_id == "B"
		structure = structure[mask]
		# save to temp-file
		receptor_file = PDBFile()
		receptor_file.set_structure(structure)
		temp_name = "/tmp/temp"+str( binascii.b2a_hex(os.urandom(15)))+".pdb"
		receptor_file.write(temp_name)
		# load again
		pdb_file = temp_name

		protein_dict = {}
		struct = fo.read_any(pdb_file)
		atoms = fo.bp_to_df(struct)
		sequence = seq.get_chain_sequences(atoms)

		protein_dict["atoms"] = atoms
		protein_dict["seq"] = sequence
		protein_dict['fitness'] = fitness

		node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(protein_dict["atoms"])
		prot_graph = Data(node_feats, edge_index, edge_feats, y=fitness, pos=pos)
		torch.save(prot_graph, f"{output_dir}/{pdb_id}.pth")

		print(f"{pdb_id} processed", flush=True)

	except Exception as e:
		print(f"Could not process {pdb_id} because of {e}", flush=True)
		traceback.print_exc()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default=".", help="Data directory")
	parser.add_argument("--dataset", default="streptavidin", help="Dataset to preprocess")
	parser.add_argument("--save_dir", default=".", help="Save directory")
	parser.add_argument("--make_lmdb", type=str2bool, default=False, help="Whether to prepare the lmdb dataset")
	args = parser.parse_args()

	base_dir = f"{args.data_dir}"
	save_dir = f"{args.save_dir}"

	filename = "../../../data/streptavidin/5sites.xls"
	loader = BaselLoader(filename)
	dts = loader.load()

	filename = "../../../data/streptavidin/2sites.xls"
	loader = BaselLoader(filename)
	total_dts = loader.load(parent='SK', positions=[112, 121])
	total_dts = loader.add_mutations('T111T+N118N+A119A', total_dts)

	total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)
	# total_dts = drop_neural_mutations(total_dts)
	total_dts = create_neural_mutations(total_dts)
	total_dts['LogFitness'] = np.log10(total_dts['Fitness'])


	if not os.path.exists(base_dir):
		raise ValueError(f"{base_dir} does not exist. Please add the dataset in this location.")

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	prepare_graphs_parallel(total_dts, base_dir, save_dir)


if __name__ == "__main__":
	print ("Welcome to preprocessing... ")
	main()
