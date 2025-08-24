import numpy as np
import pandas as pd
from os.path import join, dirname, realpath, isfile
import multiprocessing as mp
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations, generate_all_combination

from pyrosetta import *
from pyrosetta.toolbox.mutants import mutate_residue

init()

def data_dir(subdir):
    return join(dirname(realpath(__file__)), subdir, "data")

# filename = "../../../data/streptavidin/5sites.xls"
# loader = BaselLoader(filename)
# dts = loader.load()
#
# filename = "../../../data/streptavidin/2sites.xls"
# loader = BaselLoader(filename)
# total_dts = loader.load(parent = 'SK', positions = [112,121])
# total_dts = loader.add_mutations('T111T+N118N+A119A',total_dts)
#
# total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)
# total_dts = create_neural_mutations(total_dts)
#add_memb = AddMembraneMover("from_structure")
#add_memb.apply(pose)

## TODO: Relax first then mutate
def calculate_single(receptor, name, mutation, data_dest, subunit):
	# Repack and score the native conformation
	if isfile(data_dest):
		print("File exist")
	else:
		if mutation!="":
			#mut = receptor
			for i in range(4):
				for elem in mutation.split("+"):
					dest = elem[-1]
					number = int(elem[1:-1])
					print ("Mutating:",number, dest)
					mutate_residue(receptor, number-15+119*i, dest, pack_radius = 8)

			#mut = predict_ddG.mutate_residue(mut, number, dest, 8.0, sfxn)
			receptor.dump_pdb(data_dest)
		else:
			receptor.dump_pdb(data_dest)

def calculate(arg):
	name = arg[0]
	mutation = arg[1]
	calculate_single(pose, name, mutation, 'B')

def calculate_in_paralel(names, mutations, n_jobs = None):
	if n_jobs is None:
		cores = mp.cpu_count()
	else:
		cores = n_jobs
	pool = mp.Pool(cores)
	results = pool.map(calculate, [(name,mutation) for (name,mutation) in zip(names,mutations)])
	return results


if __name__ == "__main__":

	total_dts = pd.read_csv('new_mutants.csv')
	total_dts.columns = ["0","Mutation"]
	parent_struct_file = '../../data/streptavidin/6j6j-nohet.pdb'
	# file = PDBFile.read(parent_struct_file)
	# structure = file.get_structure(include_bonds = True, model=1)
	# structure = structure[structure.chain_id == "B"]
	# structure = structure[struc.filter_amino_acids(structure)]
	# receptor = structure[structure.hetero == False]

	pose = pose_from_pdb(parent_struct_file)
	force = True

	new_mutants = generate_all_combination([111,112,118,119,121], ['T','S','N','A','K'])

	cores = mp.cpu_count()-2
	pool = mp.Pool(cores)
	results = pool.map(calculate, [(name,mutation) for (name,mutation) in zip(new_mutants,new_mutants)])

