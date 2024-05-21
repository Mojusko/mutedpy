import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# feature importance
names = ["fea_imp-geometric-aa-1.csv"]+["fea_imp-geometric-aa"+str(i)+".csv" for i in range(9)]
for name in names:
	filename = "results_strep/" + name
	
	dts = pd.read_csv(filename)
	dts = dts.drop(['Unnamed: 0'], axis=1)
	dts = dts.sort_values(by=['0'],ascending=False)[0:25]
	dts.plot.bar(x='1', y='0', rot=90)
	plt.title(name)

	plt.show()

for i in range(10):
	name = "lasso-features"+str(i)+".csv"
	filename = "results_strep/" + name

	dts = pd.read_csv(filename)
	dts = dts.drop(['Unnamed: 0'], axis=1)
	dts = dts.sort_values(by=['0'],ascending=False)[0:25]
	dts.plot.bar(x='1', y='0', rot=90, color = "red")
	plt.title(str(i)+"L1")
	plt.show()

