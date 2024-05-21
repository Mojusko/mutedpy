import numpy as np
import biotite
import biotite.structure as struc
import copy
from biotite.structure.io.pdb import PDBFile
import pandas as pd
from mutedpy.utils.loaders.loader_basel import BaselLoader
import multiprocessing as mp
import miniball
from tqdm.contrib.concurrent import process_map  # or thread_map
import timeit

from os import listdir
from os.path import isfile, join

def distance_map_centre_mass(protein_structure):
	"""

	:param protein_structure:
	:return:
	"""
	centre_maps = np.zeros(shape = (biotite.structure.get_residue_count(protein_structure),3))
	for index,res in enumerate(biotite.structure.residue_iter(protein_structure)):
		centre_maps[index,:] = struc.mass_center(res)
	return centre_maps.reshape(-1)

def get_total_charge_per_aa(protein_structure):
	"""

	:param protein_structure:
	:return:
	"""
	import biotite.structure.info as info

	charge = np.zeros(biotite.structure.get_residue_count(protein_structure))
	#protein_structure_new = copy.deepcopy(protein_structure)

	#protein_structure_new.atom_name =  [a[0] if a[0].isalpha() else a[1] for a in protein_structure.atom_name]
	#protein_structure_new.add_annotation()

	for index,res in enumerate(biotite.structure.residue_iter(protein_structure)):
		charge[index] = np.nansum(biotite.structure.partial_charges(res,iteration_step_num = 8))*1e8
	#print (charge)
	return charge

def get_dihedrals(protein_structure):
	phi, psi, omega = biotite.structure.dihedral_backbone(protein_structure)
	x = np.concatenate([phi,psi,omega])
	x = x[~np.isnan(x)]
	return x

def number_of_hydrogen_bonds(protein_structure):
	h_bonds = np.zeros(biotite.structure.get_residue_count(protein_structure))
	trip = biotite.structure.hbond(protein_structure)
	hydrogens = np.unique(trip[:,1])
	unique_elemns, counts = np.unique(protein_structure[hydrogens].res_id-13, return_counts = True)
	h_bonds[unique_elemns] = counts
	return h_bonds

def sasas_per_residue_avg(protein_structure):
	sasa1 = np.zeros(biotite.structure.get_residue_count(protein_structure))
	sasa2 = np.zeros(biotite.structure.get_residue_count(protein_structure))
	sasa3 = np.zeros(biotite.structure.get_residue_count(protein_structure))
	sasa4 = np.zeros(biotite.structure.get_residue_count(protein_structure))
	sasa5 = np.zeros(biotite.structure.get_residue_count(protein_structure))
	sasa6 = np.zeros(biotite.structure.get_residue_count(protein_structure))

	residue_starts = biotite.structure.get_residue_starts(protein_structure)
	for index,res_start in enumerate(residue_starts):
		mask = biotite.structure.get_residue_masks(protein_structure,[res_start])
		sasa1[index] = np.nansum(struc.sasa(protein_structure, atom_filter = mask,point_number = 500, probe_radius = 1.4, ignore_ions = True))
		sasa2[index] = np.nansum(struc.sasa(protein_structure, atom_filter=mask, point_number=500, probe_radius=1.4, ignore_ions=False))
		sasa3[index] = np.nansum(struc.sasa(protein_structure, atom_filter=mask, point_number=500, probe_radius=3.5, ignore_ions=True))
		sasa4[index] = np.nansum(struc.sasa(protein_structure, atom_filter=mask, point_number=500, probe_radius=3.5, ignore_ions=False))
		sasa5[index] = np.nansum(struc.sasa(protein_structure, atom_filter=mask, point_number=500, probe_radius=5.5, ignore_ions=True))
		sasa6[index] = np.nansum(struc.sasa(protein_structure, atom_filter=mask, point_number=500, probe_radius=5.5, ignore_ions=False))
	return sasa1, sasa2, sasa3, sasa4, sasa5, sasa6

def sasa_per_residue_avg(protein_structure, ignore_ions = True, probe_radius = 1.4):
	sasa = np.zeros(biotite.structure.get_residue_count(protein_structure))
	residue_starts = biotite.structure.get_residue_starts(protein_structure)
	for index,res_start in enumerate(residue_starts):
		mask = biotite.structure.get_residue_masks(protein_structure,[res_start])
		sasa[index] = np.nansum(struc.sasa(protein_structure,atom_filter = mask,point_number = 1000,
										   probe_radius = probe_radius, ignore_ions = ignore_ions))
	return sasa

def sasa_per_residue_residue(protein_structure, ignore_ions = True, probe_radius = 1.4):
	sasa = np.zeros(biotite.structure.get_residue_count(protein_structure))
	for index,res in enumerate(biotite.structure.residue_iter(protein_structure)):
		sasa[index] = np.nansum(struc.sasa(res,point_number = 1000,
										   probe_radius = probe_radius, ignore_ions = ignore_ions))
	return sasa

def mini_balls(protein_structure):
	centers = np.zeros(shape = (biotite.structure.get_residue_count(protein_structure),3))
	radiuses = np.zeros(biotite.structure.get_residue_count(protein_structure))
	for index,res in enumerate(biotite.structure.residue_iter(protein_structure)):
		coords = biotite.structure.coord(res)
		centers[index,:], radiuses[index] = miniball.get_bounding_ball(coords)
	return centers.reshape(-1), radiuses

def size_of_residue(protein_structure):
	size = np.zeros(biotite.structure.get_residue_count(protein_structure))
	for index,res in enumerate(biotite.structure.residue_iter(protein_structure)):
		H, edges = biotite.structure.density(res)
		#size[index] = np.array(edges)
		print (edges)
	return size

def structurally_close(str1, str2, threshold = 2):
	print ("Comparing:")
	print (str1)
	print (str2)
	if len(set(str1).symmetric_difference(str2))<=threshold:
		return True
	else:
		return False

if __name__ == "__main__":

	filename = "../../../data/streptavidin/5sites.xls"
	loader = BaselLoader(filename)
	dts = loader.load()
	filename = "../../../data/streptavidin/2sites.xls"
	loader = BaselLoader(filename)
	total_dts = loader.load(parent='SK', positions=[112, 121])
	total_dts = loader.add_mutations('T111T+N118N+A119A', total_dts)
	total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)

	total_dts = total_dts['Mutation']

	#print (total_dts)
	df = pd.read_csv("../../structure_generation/new_mutants.csv")
	df.drop(columns=df.columns[0],
			axis=1,
			inplace=True)
	df.columns = ['Mutation']
	df = df['Mutation']

	df2 = pd.read_csv('../active_learning/Geometric/AA_model_sorted_by_std.csv')
	df2 = df2['Mutation']
	total_dts = pd.concat([total_dts,df,df2], ignore_index=True, sort=False)

	total_dts.to_csv("to-copy.csv", header = False, index = False)
	print (total_dts)

	mutations = total_dts.unique().tolist()

	d = {'mutation': [], 'charges': [], 'distance' :[], 'surf1':[],'surf2':[],
		 'surf3':[],'surf4':[],'surf5':[],'surf6':[], 'no_hbonds':[], 'dihedrals':[],
		 'centers':[],'radiuses':[]	 }

	counter = 0
	d2 = {}
	def eval_params(elem):
		try:
			print(elem)
			struct_file = '../../../data/streptavidin/rosetta_selection/' + elem + '.pdb'
			file = PDBFile.read(struct_file)
			structure = file.get_structure(include_bonds=True, model=1)

			mask = np.logical_or(structure.chain_id == "B",structure.chain_id == "C")
			structure = structure[mask]
			receptor = structure

			distance = distance_map_centre_mass(receptor)
			surf1 = sasa_per_residue_avg(receptor)
			surf2 = sasa_per_residue_avg(receptor, ignore_ions=False)
			surf3 = sasa_per_residue_avg(receptor, probe_radius=3.5)
			surf4 = sasa_per_residue_avg(receptor, ignore_ions=False, probe_radius=3.5)
			surf5 = sasa_per_residue_avg(receptor, probe_radius=5.5)
			surf6 = sasa_per_residue_avg(receptor, ignore_ions=False, probe_radius=5.5)
			centers, radiuses = mini_balls(receptor)
			return [elem, distance, surf1, surf2, surf3, surf4, surf5,surf6, centers, radiuses]
		except:
			print ("Failed.")


	debug = False
	# debug
	if debug:
		mutations = ["T111P+S112K+N118P+A119A+K121G",mutations[1]]
		cores = 1
	else:
		cores = mp.cpu_count()-1

	results = process_map(eval_params,[elem for elem in mutations], max_workers=cores)

	#pool = mp.Pool(cores)
	#results_strep = pool.map(eval_params,[elem for elem in mutations])

	for index, _ in enumerate(mutations):
		elem, distance, surf1, surf2, surf3, surf4, surf5,surf6, centers, radiuses = results[index]
		d['mutation'].append(elem)
		d['distance'].append(distance)

		d['centers'].append(centers)
		d['radiuses'].append(radiuses)

		d['surf1'].append(surf1)
		d['surf2'].append(surf2)
		d['surf3'].append(surf3)
		d['surf4'].append(surf4)
		d['surf5'].append(surf5)
		d['surf6'].append(surf6)

	data = np.concatenate((np.array(d['distance']),
						   np.array(d['centers']),
						   np.array(d['radiuses']),
						   np.array(d['surf1']),
						   np.array(d['surf2']),
						   np.array(d['surf3']),
						   np.array(d['surf4']),
						   np.array(d['surf5']),
						   np.array(d['surf6'])
						   		   ), axis =1)

	sizes = [d['distance'][0].shape[0],
			 d['centers'][0].shape[0], d['radiuses'][0].shape[0],
			 d['surf1'][0].shape[0],d['surf2'][0].shape[0],d['surf3'][0].shape[0],
			 d['surf4'][0].shape[0],d['surf5'][0].shape[0],d['surf6'][0].shape[0]
			]
	names = ['cm','center','radius','surf1','surf2','surf3','surf4','surf5','surf6']

	columns_blocks = [[name +"_"+ str(j) for j in range(size) ]for (size,name) in zip(sizes,names)]
	columns = [item for sublist in columns_blocks for item in sublist]
	print (columns)
	dts = pd.DataFrame(data=data, columns=columns)
	dts['mutation'] = d['mutation']
	if not debug:
		dts.to_csv("../../../data/streptavidin/new_features_rosetta_small.csv")
		dts.to_csv("../../protein_learning/data/new_features_rosetta_small.csv")
		dts.to_csv("new_features_rosetta_small.csv")

	print (dts)

