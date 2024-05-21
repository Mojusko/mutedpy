from mutedpy.utils.sequences.sequence_utils import generate_random_mutations, generate_all_combination
from biotite.structure.io.pdb import PDBFile
from biotite.application.foldx import FoldXApp

import pandas as pd
from mutedpy.utils.loaders.loader_basel import BaselLoader
from os.path import join, dirname, realpath, isfile
import biotite.structure as struc
import multiprocessing as mp
#

filename = "../../data/streptavidin/5sites.xls"
loader = BaselLoader(filename)
dts = loader.load()

filename = "../../data/streptavidin/2sites.xls"
loader = BaselLoader(filename)
total_dts = loader.load(parent = 'SK', positions = [112,121])
total_dts = loader.add_mutations('T111T+N118N+A119A',total_dts)

total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)


parent_struct_file = '../../data/streptavidin/6j6j.pdb'
file = PDBFile.read(parent_struct_file)
structure = file.get_structure(include_bonds = True, model=1)
#structure = structure[structure.chain_id == "B"]
structure = structure[struc.filter_amino_acids(structure)]
receptor = structure[structure.hetero == False]

force = False
N = 1000
new_mutants = generate_random_mutations(N, [111,112,118,119,121], ['T','S','N','A','K'],prior_muts=total_dts['Mutation'])
#new_mutants = generate_all_combination([111,112,118,119,121], ['T','S','N','A','K'])
new_mutants = new_mutants + total_dts['Mutation'].tolist()


dt = pd.DataFrame(new_mutants)
dt.to_csv('new_mutants.csv')


def calculate_single(receptor, name, mutation, subunit):
	data_dest = '../../data/streptavidin/foldx/' + name + '.pdb'
	if isfile(data_dest) and force:
		print("File exist")
	else:
		print ("mutation:", mutation)
		if mutation!="":
			try:
				mutant = receptor.copy()
				for sub in ["A","B","C","D"]:
					app = FoldXApp(mutant, mutation, subunit=sub)
					app.start()
					app.join()
					mutant = app.get_mutant()

				receptor_file = PDBFile()
				receptor_file.set_structure(mutant)
				receptor_file.write(data_dest)
			except:
				print ("Failure:", mutation)
		else:
			receptor_file = PDBFile()
			receptor_file.set_structure(receptor)
			receptor_file.write(data_dest)

def calculate(arg):
	name = arg[0]
	mutation = arg[1]
	calculate_single(receptor, name, mutation, 'B')

cores = mp.cpu_count()//2
pool = mp.Pool(cores)
results = pool.map(calculate, [(name,mutation) for (name,mutation) in zip(new_mutants,new_mutants)])



