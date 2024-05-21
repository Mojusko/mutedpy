import numpy as np
import torch
from stpy.helpers.abitrary_sampling import randomly_split_set_without_duplicates
from stpy.helpers.abitrary_sampling import randomly_split_set_without_duplicates_balanced
import pickle

def create_subsampled_splits(x,y,no_splits, n_subsample, n_test, file_to_save):

	if n_subsample < n_test:
		raise AssertionError("Number of subsampled data needs to be larger than test")
	else:
		indices = np.random.choice(np.arange(0, x.size()[0], 1), n_subsample, replace=False)
		x_sub = x[indices,:]
		y_sub = y[indices,:]
		return create_splits(x_sub, y_sub, no_splits, n_test, file_to_save)


def create_random_subsampled_splits(x,y,no_splits, n_subsample, n_test, file_to_save):

	if n_subsample < n_test:
		raise AssertionError("Number of subsampled data needs to be larger than test")
	else:
		d = {}
		for split in range(no_splits):
			indices = np.random.choice(np.arange(0, x.size()[0], 1), n_subsample, replace=False)
			x_sub = x[indices,:]
			y_sub = y[indices,:]
			d[split] = {}

			mask_test, mask_train = randomly_split_set_without_duplicates(x_sub, size=n_test)
			x_test = x_sub[mask_test,:]
			y_test = y_sub[mask_test,:]

			# train
			x_train = x_sub[mask_train,:]
			y_train = y_sub[mask_train,:]

			d[split]['x'] = {'test':x_test,'train':x_train}
			d[split]['y'] = {'test': y_test, 'train': y_train}

		with open(file_to_save, "wb") as f:
			pickle.dump(d,f)
		return d



def create_splits(x,
				  y,
				  no_splits,
				  n_test,
				  file_to_save,
				  stratification = None
				  ):
	d = {}

	for i in range(no_splits):
		d[i] = {}
		if stratification == None:
			mask_test, mask_train = randomly_split_set_without_duplicates(x, size=n_test)

		elif stratification == "balanced":
			# balances high and low fitness in train and test
			mask_test, mask_train = randomly_split_set_without_duplicates_balanced(x, y, size=n_test, max_bins = 10)

		elif stratification == "polarized":
			raise NotImplementedError("Given stratification is not implemented.")
			# creates a polarity where high performing are in test and low in train

		else:
			raise NotImplementedError("Given stratification is not implemented.")
		# test
		x_test = x[mask_test,:]
		y_test = y[mask_test,:]

		# train
		x_train = x[mask_train,:]
		y_train = y[mask_train,:]

		d[i]['x'] = {'test':x_test,'train':x_train}
		d[i]['y'] = {'test': y_test, 'train': y_train}

	with open(file_to_save, "wb") as f:
		pickle.dump(d,f)
	return d
def load_splits(no_splits, file_to_load):
	output = []
	with open(file_to_load, "rb") as f:
		d = pickle.load(f)
		max_number_of_splits = max(d.keys())
		for i in range(min(no_splits,max_number_of_splits)):
			output.append([d[i]['x']['test'],d[i]['y']['test'],d[i]['x']['train'],d[i]['y']['train']])
		return output

if __name__ == "__main__":


	pass
