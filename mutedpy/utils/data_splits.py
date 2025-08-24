import numpy as np
import torch
from stpy.helpers.abitrary_sampling import randomly_split_set_without_duplicates, randomly_split_set_without_duplicates_general
import json


def split_data_names_save_to_json(name, names, n_test = 150, n_validaiton = 150):
	unique_names = names.unique()
	permuted_unique_names = np.random.permutation(unique_names)
	test_names = permuted_unique_names[0:n_test]
	validation_names = permuted_unique_names[n_test:n_test+n_validaiton]
	train_names = permuted_unique_names[n_test+n_validaiton:]
	d = {}
	d['test'] = test_names.tolist()
	d['train'] = train_names.tolist()
	d['val'] = validation_names.tolist()

	with open(name, "w") as f:
		json.dump(d,f,indent=4)

def split_data_without_duplicates(x, n_test=150, n_validation = None):
	n = x.size()[0]

	if n_validation is None:
		mask_test, mask_train = randomly_split_set_without_duplicates(x, size=n_test)
		return mask_test, mask_train
	else:
		mask_test, mask_val, mask_train = randomly_split_set_without_duplicates_general(x, sizes = [n_test, n_validation, 100000000])
		return mask_test, mask_val, mask_train


def generate_save_save_json_splits_mask(x, name, n_test=150, n_validation = 150, modes = ["test", "val", "train"]):
	masks = split_data_without_duplicates(x, n_test = n_test, n_validation = n_validation)
	m = np.arange(0,x.size()[0],1)
	d = {}
	for index,mode in enumerate(modes):
		d[mode] = m[masks[index]].tolist()
	with open(name, "w") as f:
		json.dump(d,f,indent=4)

def load_json_splits_mask(name, modes = ["test","val","train"]):
	with open(name) as f:
		d = json.load(f)
	n = 0
	for index, mode in enumerate(modes):
		n = np.maximum(np.max(np.array(d[mode])),n)
	masks = []
	n = n + 1
	for index, mode in enumerate(modes):
		mask = torch.zeros(size = (n,1)).view(-1).bool()
		mask[d[mode]] = 1
		masks.append(mask)
	return masks

if __name__ == "__main__":
	x = torch.Tensor([[2, 1, 1], [2, 1, 1], [2, 2, 2], [3, 2, 2], [2, 1, 1], [4, 2, 1], [4, 2, 4]]).double()
	name = "test.json"
	generate_save_save_json_splits(x,name, n_test=1, n_validation=2)
	masks = load_json_splits(name)

	for mask in masks:
		print (mask)
