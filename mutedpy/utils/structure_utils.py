import pickle
import os
import binascii
import numpy as np
import torch
import biotite
import biotite.structure as struc

from biotite.structure.io.pdb import PDBFile
from stpy.helpers.helper import cartesian
import pandas as pd
from mutedpy.utils.loaders.loader_basel import BaselLoader
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations, order_mutations, add_variant_column
import multiprocessing as mp
import miniball
from cvxpylayers.torch import CvxpyLayer

import errno
import os
import signal
import functools
from os import listdir
from os.path import isfile, join

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

def get_parent_aa_from_pdb(pdb_file_name, positions):
	out = []
	pdb_file = biotite.structure.io.pdb.PDBFile.read(pdb_file_name)
	pdb = biotite.structure.io.pdb.get_structure(pdb_file)[0]
	resids, resnames = biotite.structure.get_residues(pdb)
	for position in positions:
		index = np.argmax(resids == position)
		out.append(resnames[index])
	return out


def distance_map_centre_mass(protein_structure):
	structure = protein_structure[protein_structure.hetero == False]
	centre_maps = np.zeros(shape = (biotite.structure.get_residue_count(structure),3))
	for index,res in enumerate(biotite.structure.residue_iter(structure)):
		centre_maps[index,:] = struc.mass_center(res)
	return centre_maps.reshape(-1)

def get_total_charge_per_aa(protein_structure):
	structure = protein_structure[protein_structure.hetero == False]
	charge = np.zeros(biotite.structure.get_residue_count(structure))
	for index,res in enumerate(biotite.structure.residue_iter(structure)):
		charge[index] = np.nansum(biotite.structure.partial_charges(res,iteration_step_num = 8))*1e8
	return charge

def get_dihedrals(protein_structure):
	structure = protein_structure[protein_structure.hetero == False]
	phi, psi, omega = biotite.structure.dihedral_backbone(structure)
	x = np.concatenate([phi,psi,omega])
	x = x[~np.isnan(x)]
	return x

def number_of_hydrogen_bonds(protein_structure):
	structure = protein_structure[protein_structure.hetero == False]
	h_bonds = np.zeros(biotite.structure.get_residue_count(structure))
	trip = biotite.structure.hbond(structure)
	hydrogens = np.unique(trip[:,1])
	unique_elemns, counts = np.unique(structure[hydrogens].res_id-13, return_counts = True)
	h_bonds[unique_elemns] = counts
	return h_bonds

def sasa_per_residue_avg(protein_structure, ignore_ions = True, probe_radius = 1.4):
	structure = protein_structure[protein_structure.hetero == False]
	sasa = np.zeros(biotite.structure.get_residue_count(structure))
	residue_starts = biotite.structure.get_residue_starts(structure)
	for index,res_start in enumerate(residue_starts):
		mask = biotite.structure.get_residue_masks(structure,[res_start])
		sasa[index] = np.nansum(struc.sasa(structure,atom_filter = mask,point_number = 1000,
										   probe_radius = probe_radius, ignore_ions = ignore_ions))
	return sasa

def sasa_per_residue_residue(protein_structure, ignore_ions = True, probe_radius = 1.4):
	structure = protein_structure[protein_structure.hetero == False]
	sasa = np.zeros(biotite.structure.get_residue_count(structure))
	for index,res in enumerate(biotite.structure.residue_iter(structure)):
		sasa[index] = np.nansum(struc.sasa(res,point_number = 1000,
										   probe_radius = probe_radius, ignore_ions = ignore_ions))
	return sasa

def mini_balls(protein_structure):
	structure = protein_structure[protein_structure.hetero == False]
	centers = np.zeros(shape = (biotite.structure.get_residue_count(structure),3))
	radiuses = np.zeros(biotite.structure.get_residue_count(structure))
	for index,res in enumerate(biotite.structure.residue_iter(structure)):
		coords = biotite.structure.coord(res)
		centers[index,:], radiuses[index] = miniball.get_bounding_ball(coords)
	return centers.reshape(-1), radiuses

def size_of_residue(protein_structure):
	structure = protein_structure[protein_structure.hetero == False]
	size = np.zeros(biotite.structure.get_residue_count(structure))
	for index,res in enumerate(biotite.structure.residue_iter(structure)):
		H, edges = biotite.structure.density(res)
		#size[index] = np.array(edges)
		print (edges)


def structurally_close(str1, str2, threshold = 2):
	print ("Comparing:")
	print (str1)
	print (str2)
	if len(set(str1).symmetric_difference(str2))<=threshold:
		return True
	else:
		return False

@timeout(50,"Timed out")
def kwfinder(name):

	results = pyKVFinder.run_workflow(name, include_depth=True, include_hydropathy=True, hydrophobicity_scale='EisenbergWeiss', nthreads = 1)

	print (results.volume)
	print (results.area)
	print (results.avg_depth)
	print (results.max_depth)
	print (results.avg_hydropathy)

	# ids_of_cavities = []
	#
	# print (cavities)
	#
	# for cavity in cavities.keys():
	# 	resnum = [int(a[0]) for a in cavities[cavity]]
	#
	# 	detected = False
	# 	for index,detected_cavity in enumerate(detected_cavities):
	# 		resnum_det = [a for a in detected_cavity]
	#
	# 		if structurally_close(resnum, resnum_det):
	# 			detected = True
	# 			ids_of_cavities.append(index)
	# 			print ("Match found.")
	#
	# 			break
	# 	if detected == False:
	# 		detected_cavities.append(resnum)
	# 		ids_of_cavities.append(len(detected_cavities)-1)
	# 		print ("Added new cavity; not recongized before.")
	#
	# 	print (ids_of_cavities)

	return [results.volume, results.area, results.avg_depth, results.max_depth, results.avg_hydropathy,results.residues]

detected_cavities = []

if __name__ == "__main__":

	filename = "../../data/streptavidin/5sites.xls"
	loader = BaselLoader(filename)
	dts = loader.load()
	filename = "../../data/streptavidin/2sites.xls"
	loader = BaselLoader(filename)
	total_dts = loader.load(parent='SK', positions=[112, 121])
	total_dts = loader.add_mutations('T111T+N118N+A119A', total_dts)
	total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)

	total_dts = total_dts['Mutation']

	#print (total_dts)
	df = pd.read_csv("../structure-generation/new_mutants.csv")
	df.drop(columns=df.columns[0],
			axis=1,
			inplace=True)
	df.columns = ['Mutation']
	df = df['Mutation']
	#total_dts = pd.concat([total_dts,df], ignore_index=True, sort=False)


	# list of files in rosetta folder
	first_mutants = df.unique().tolist()
	mutations = [f.split(".")[0] for f in listdir("../../data/streptavidin/rosetta/") if isfile(join("../../data/streptavidin/rosetta/", f))]
	#df = pd.DataFrame(data = files, columns=['Mutation'])

	#mutations = .unique()
	indices = np.random.choice(np.arange(0, len(mutations), 1), 200000, replace = False)
	mutations = first_mutants + [mutations[i] for i in indices]
	print (len(mutations))

	d = {'mutation': [], 'charges': [], 'distance' :[], 'surf1':[],'surf2':[],
		 'surf3':[],'surf4':[],'surf5':[],'surf6':[], 'no_hbonds':[], 'dihedrals':[],
		 'centers':[],'radiuses':[]	 }

	counter = 0
	d2 = {}
	def eval_params(elem):
		try:
			print(elem)
			struct_file = '../../data/streptavidin/rosetta/' + elem + '.pdb'
			file = PDBFile.read(struct_file)
			structure = file.get_structure(include_bonds=True, model=1)

			mask = np.logical_or(structure.chain_id == "B",structure.chain_id == "C")
			structure = structure[mask]
			receptor = structure

			charges = get_total_charge_per_aa(receptor)
			distance = distance_map_centre_mass(receptor)

			surf1 = sasa_per_residue_avg(receptor)
			surf2 = sasa_per_residue_avg(receptor, ignore_ions=False)
			surf3 = sasa_per_residue_avg(receptor, probe_radius=3.5)
			surf4 = sasa_per_residue_avg(receptor, ignore_ions=False, probe_radius=3.5)
			surf5 = sasa_per_residue_avg(receptor, probe_radius=5.5)
			surf6 = sasa_per_residue_avg(receptor, ignore_ions=False, probe_radius=5.5)
			no_hbonds = number_of_hydrogen_bonds(receptor)
			dihedrals = get_dihedrals(receptor)
			centers, radiuses = mini_balls(receptor)

			# create  dimer
			temp = PDBFile()
			temp.set_structure(structure)
			name = binascii.b2a_hex(os.urandom(15)).decode("utf-8")
			temp.write("/tmp/"+name+".pdb")
			kwdata = kwfinder("/tmp/"+name+".pdb")
			del temp
			os.remove("/tmp/"+name+".pdb")

			return [elem, charges, distance, surf1, surf2, surf3, surf4, surf5,surf6, no_hbonds, dihedrals, centers, radiuses, kwdata]
		except:
			print ("Failed.")


	debug = False
	# debug
	if debug:
		mutations = ["T111P+S112K+N118P+A119A+K121G",mutations[1]]
		cores = 1
	else:
		cores = mp.cpu_count()
	pool = mp.Pool(cores)

	results = pool.map(eval_params,[elem for elem in mutations])

	for index, _ in enumerate(mutations):
		elem, charges, distance, surf1, surf2, surf3, surf4, surf5,surf6, no_hbonds, dihedrals, centers, radiuses, kwdata = results[index]
		d['mutation'].append(elem)
		d['charges'].append(charges)
		d['distance'].append(distance)

		d['centers'].append(centers)
		d['radiuses'].append(radiuses)

		d['surf1'].append(surf1)
		d['surf2'].append(surf2)
		d['surf3'].append(surf3)
		d['surf4'].append(surf4)
		d['surf5'].append(surf5)
		d['surf6'].append(surf6)
		d['no_hbonds'].append(no_hbonds)
		d['dihedrals'].append(dihedrals)
		d2[elem] = kwdata

	data = np.concatenate((np.array(d['charges']),
						   np.array(d['distance']),
						   np.array(d['centers']),
						   np.array(d['radiuses']),
						   np.array(d['surf1']),
						   np.array(d['surf2']),
						   np.array(d['surf3']),
						   np.array(d['surf4']),
						   np.array(d['surf5']),
						   np.array(d['surf6']),
						   np.array(d['no_hbonds']),
						   np.array(d['dihedrals'])), axis =1)

	sizes = [d['charges'][0].shape[0],d['distance'][0].shape[0],
			 d['centers'][0].shape[0], d['radiuses'][0].shape[0],
			 d['surf1'][0].shape[0],d['surf2'][0].shape[0],d['surf3'][0].shape[0],
			 d['surf4'][0].shape[0],d['surf5'][0].shape[0],d['surf6'][0].shape[0],
			 d['no_hbonds'][0].shape[0],d['dihedrals'][0].shape[0]]
	names = ['charge','cm','center','radius','surf1','surf2','surf3','surf4','surf5','surf6','n_hbonds','dihedrals']

	columns_blocks = [[name +"_"+ str(j) for j in range(size) ]for (size,name) in zip(sizes,names)]
	columns = [item for sublist in columns_blocks for item in sublist]
	print (columns)
	dts = pd.DataFrame(data=data, columns=columns)
	dts['mutation'] = d['mutation']
	if not debug:
		dts.to_hdf("../../data/streptavidin/new_features_rosetta_large.hdf")
		dts.to_hdf("../protein_learning/data/new_features_rosetta_large.hdf")
		dts.to_hdf("new_features_rosetta_large.hdf")

		with open('cavities.pkl', 'wb') as f:
			pickle.dump(d2, f)

	print (dts)

