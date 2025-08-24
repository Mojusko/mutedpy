import torch
import numpy as np
import pandas as pd
import pickle
with open('cavities.pkl', 'rb') as f:
	dict = pickle.load(f)

mutations = dict.keys()

names = ["volume","area","avg_deph","max_depth","avg_hydropathy"]
detected_cavities = []
def structurally_close(str1, str2, threshold = 3):
	print ("Comparing:")
	print (str1)
	print (str2)

	if len(set(str1).symmetric_difference(str2))<=threshold:
		return True
	else:
		return False

ids = []
no_cavity_types = 0
for mut in mutations:
	ids_of_cavities = []
	print ("analyzing", mut)
	res = dict[mut]
	[volume, area, avg_depth, max_depth, avg_hydropathy, residues] = res
	print (residues)
	for cavity in residues.keys():
		resnum = [a[0]+a[1] for a in residues[cavity]]

		detected = False
		for index,detected_cavity in enumerate(detected_cavities):
			resnum_det = [a for a in detected_cavity]

			if structurally_close(resnum, resnum_det):
				detected = True
				ids_of_cavities.append(index)
				print ("Match found.")

				break
		if detected == False:
			detected_cavities.append(resnum)
			ids_of_cavities.append(len(detected_cavities)-1)
			print ("Added new cavity; not recongized before.")
	no_cavity_types = max(no_cavity_types,max(ids_of_cavities))
	ids.append(ids_of_cavities)
	print (ids_of_cavities)

no_features_classes = len(names)
features = np.zeros(shape = (len(mutations),no_features_classes*(no_cavity_types+1)))
d = {'Mutation':[]}
for index,mut in enumerate(mutations):
	print ("Working on:",mut)
	res = dict[mut]
	d['Mutation'].append(mut)
	id = ids[index]
	[volume, area, avg_depth, max_depth, avg_hydropathy, residues] = res
	raw = [volume, area, avg_depth, max_depth, avg_hydropathy]
	for cavity_id, cavity in enumerate(residues.keys()):
		print ("cavity id",cavity_id, "cavity", cavity)
		for j in range(no_features_classes):
			features[index,j*no_cavity_types+id[cavity_id]] = raw[j][cavity]
		print (features[index,:])

columns_blocks = [[name +"_"+ str(j) for j in range(no_cavity_types+1)] for name in names]
columns = [item for sublist in columns_blocks for item in sublist]
print (len(columns))
print (features.shape)
dts = pd.DataFrame(data=features, columns=columns)
print (dts)
dts['mutation'] = d['Mutation']
dts.to_csv("../../data/streptavidin/new_features_volume.csv")
dts.to_csv("../protein_learning/data/new_features_volume.csv")
dts.to_csv("new_features_volume.csv")





