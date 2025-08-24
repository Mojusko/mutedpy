from mutedpy.utils.loaders.loader import Loader
from mutedpy.utils.sequences.sequence_utils import order_mutation_string
import pandas as pd
import xlrd
import numpy as np


class BaselLoader(Loader):

	def __init__(self, filename):
		self.filename = filename

	def load(self, positions = [111,112,118,119,121], parent = 'TSNAK'):
		self.parent = list(parent)
		workbook = xlrd.open_workbook(self.filename)
		sheet_names = workbook.sheet_names()
		name_first = sheet_names[0]
		dts = pd.read_excel(open(self.filename, 'rb'),  sheet_name=name_first).dropna()
		positions = [str(a) for a in positions]
		dts['Mutation'] = dts['variant'].apply(lambda s: "+".join(a+b+c for a,b,c in zip(self.parent,positions,list(str(s)))  ))
		dts['Fitness']  = dts['norm_TSNAK']
		return dts

	def add_mutations(self, mutations, dts):
		f = lambda x: order_mutation_string(x+"+"+mutations)
		g = lambda x: "T"+x[0]+"NA"+x[1]
		dts["Mutation"]=dts['Mutation'].apply(f)
		dts['variant']=dts['variant'].apply(g)
		return dts



if __name__ == "__main__":

	filename = "../../../data/streptavidin/5sites.xls"
	loader = BaselLoader(filename)
	dts = loader.load()


