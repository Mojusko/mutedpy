from mutedpy.utils.loaders.loader import Loader
#from mutedpy.utils.sequence_utils import
import pandas as pd
import xlrd
import numpy as np
from mutedpy.utils.sequences.sequence_utils import order_mutations, order_mutation_string

class ZHAWLoader(Loader):

	def __init__(self, filename, single=True):
		self.filename = filename
		self.single = single

	def load(self, controls = "default"):

		workbook = xlrd.open_workbook(self.filename)
		sheet_names = workbook.sheet_names()
		datasets = []



		if controls == "default":
			pos_controls = [["E6","F7","G8"] for _ in range(len(sheet_names))]
		else:
			arr = pd.read_csv(controls, delim_whitespace = True)['pos'].values
			pos_controls = [a.split(",") for a in list(arr)]


		for index, name in enumerate(sheet_names):
			worksheet = workbook.sheet_by_name(name)

			try:
				parent = "+".join(str(worksheet.cell(0, 6).value).split("/"))
			except:
				parent = None

			if self.single == True:
				locations = [str(worksheet.cell(1, 0).value)]
			else:
				locations = str(worksheet.cell(1, 0).value).split("/")


			dts = pd.read_excel(self.filename, skiprows=2, sheet_name=name)
			dts['Mutation'] = dts['Mutation'].astype(str).str.strip("#")

			frameshift_mask = dts['Mutation'] == "FS"
			missing_mask = dts['Mutation'].astype(str) == 'nan'  # .str.isspace()
			alpha_num = ~dts['Mutation'].astype(str).str.isalpha()

			mask = ~frameshift_mask & ~missing_mask & ~alpha_num

			new_dts = dts[mask].copy(deep=True)
			new_dts['Activity norm/control'] = new_dts['Activity norm']

			del new_dts['Activity norm']

			if parent is not None:
				new_dts['control units'] = order_mutation_string(parent)
			else:
				new_dts['control units'] = "PETase"

			del new_dts['Unnamed: 0']
			del new_dts['Unnamed: 5']
			try:
				del new_dts['Unnamed: 6']
			except:
				pass

			# activity, # activity norm, # Mutation #Off-target
			if parent is not None:
				func = lambda loc: "+".join([ str(a)+str(b) for a,b, in zip(locations,loc) ]) + "+" + parent
			else:
				func = lambda loc: "+".join([str(a) + str(b) for a, b, in zip(locations, loc)])

			new_dts['Mutation'] = new_dts['Mutation'].apply(func)
			new_dts['Mutation'] = new_dts['Mutation']

			if self.single == False:
				new_dts['Plate'] = name + " comb"
			else:
				new_dts['Plate'] = name

			# add note that they are control
			func_control = lambda x: "control" if str(x).strip("\n") in pos_controls[index] else ""
			new_dts["note"] = dts['Unnamed: 0'].apply(func_control)

			# add off-targets
			func_off = lambda loc: "+".join([a.strip("\n") for a in str(loc).strip('][').split(' ')])
			new_dts['Offtarget'] = new_dts['Offtarget'].apply(func_off)
			datasets.append(new_dts)

			for index, row in new_dts.iterrows():
				if row['Offtarget'] != "":
					new_dts['Mutation'][index] = row['Offtarget'].replace('\'','') + "+" + row['Mutation']

		return pd.concat(datasets, ignore_index=True, sort=False)

if __name__ == "__main__":

	filename = "../../../data/petase/2020/1st_single_site.xls"
	loader = ZHAWLoader(filename , single = True)
	dts = loader.load(controls = "../../../data/petase/2020/controls.csv")
	print (dts['Mutation'])

