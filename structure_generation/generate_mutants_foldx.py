from biotite.structure.io.pdbqt import PDBQTFile
from biotite.structure.io.pdb import PDBFile
from biotite.application.foldx import FoldXApp
import numpy as np
import pandas as pd
from mutedpy.utils.loaders.loader_basel import BaselLoader
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations, order_mutations, create_neural_mutations
import biotite.structure.io.mmtf as mmtf
from os.path import join, dirname, realpath, isfile
import biotite.structure as struc
import multiprocessing as mp


def data_dir(subdir):
    return join(dirname(realpath(__file__)), subdir, "data")

filename = "../../data/streptavidin/5sites.xls"
loader = BaselLoader(filename)
dts = loader.load()

filename = "../../data/streptavidin/2sites.xls"
loader = BaselLoader(filename)
total_dts = loader.load(parent = 'SK', positions = [112,121])
total_dts = loader.add_mutations('T111T+N118N+A119A',total_dts)

total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)
#total_dts = drop_neural_mutations(total_dts)
total_dts = create_neural_mutations(total_dts)
total_dts['LogFitness'] = np.log10(total_dts['Fitness'])


parent_struct_file = '../../data/streptavidin/2rtg.pdb'
file = PDBFile.read(parent_struct_file)
structure = file.get_structure(include_bonds = True, model=1)
structure = structure[structure.chain_id == "B"]
structure = structure[struc.filter_amino_acids(structure)]
receptor = structure[structure.hetero == False]


print (len(total_dts['Mutation_n']))

def calculate_single(receptor, name, mutation, subunit):
	data_dest = '../../data/streptavidin/foldx/' + name + '.pdb'
	if isfile(data_dest):
		print("File exist")
	else:
		if mutation!="":
			app = FoldXApp(receptor, mutation, subunit=subunit)
			app.start()
			app.join()
			mutant = app.get_mutant()
			receptor_file = PDBFile()
			receptor_file.set_structure(mutant)
			receptor_file.write(data_dest)
		else:
			receptor_file = PDBFile()
			receptor_file.set_structure(receptor)
			receptor_file.write(data_dest)

def calculate(arg):
	name = arg[0]
	mutation = arg[1]
	calculate_single(receptor, name, mutation, 'B')

def calculate_in_paralel(names, mutations, n_jobs = None):
	if n_jobs is None:
		cores = mp.cpu_count()
	else:
		cores = n_jobs
	pool = mp.Pool(cores)
	results = pool.map(calculate, [(name,mutation) for (name,mutation) in zip(names,mutations)])
	return results

calculate_in_paralel(total_dts['Mutation'],total_dts['Mutation_n'])
