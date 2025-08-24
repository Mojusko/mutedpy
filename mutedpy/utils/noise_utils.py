import pandas as pd 
import numpy as np 
from typing import Union

def get_noise_std(total_dts: pd.DataFrame, truncation:Union[float,None] = None):
	repeated = total_dts['Mutation'].value_counts().head(30)
	total_dts_selected = total_dts
	means = []
	medians = []
	for index in repeated.index:
		total_dts_selected[total_dts_selected['Mutation']==index]
		vals = total_dts_selected[total_dts_selected['Mutation']==index]['LogFitness']
		mean = np.mean(vals)
		median = np.median(vals)
		means.append(mean)
		medians.append(median)
	residuals_mean = []
	residuals_median = []
	for index in repeated.index:
		total_dts_selected[total_dts_selected['Mutation']==index]
		vals = total_dts_selected[total_dts_selected['Mutation']==index]['LogFitness']
		mean = np.mean(vals)
		median = np.median(vals)
		residuals_mean.append((vals.values-mean))
		residuals_median.append((vals.values-median))
	
	residuals_mean = np.concatenate(residuals_mean)
	residuals_median = np.concatenate(residuals_median)

	
	if truncation is not None:
		residuals_mean_trunc = residuals_mean[np.abs(residuals_mean)<truncation]
		sigma_std = np.std(residuals_mean_trunc)
	else:
		sigma_std = np.std(residuals_mean)
		
	return sigma_std
