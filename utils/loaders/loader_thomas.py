from mutedpy.utils.loaders.loader import Loader
from mutedpy.utils.sequences.sequence_utils import order_mutation_string
import pandas as pd
import xlrd
import numpy as np

class ThomasLoader(Loader):

    def __init__(self, filename):
        super().__init__(filename)
        self.filename = filename

    def load(self, parent=''):
        self.parent = list(parent)
        workbook = xlrd.open_workbook(self.filename)

        sheet_names = workbook.sheet_names()
        dts_multiple = []
        for sheet in sheet_names[1:]:
            dts = pd.read_excel(open(self.filename, 'rb'), sheet_name=sheet)
            dts = dts[['ReadCount','PlatePosition','Mutations','WtSlope','VariantSlope']]
            dts['Plate'] = sheet
            dts['Activity'] = dts['WtSlope'].combine_first(dts['VariantSlope'])
            dts['Mutation'] = dts['Mutations'].apply(lambda x: "+".join(x.split(" ")))
            dts.loc[dts['Mutation'] == 'WT','Mutation'] = ''
            dts['Position'] = dts['Mutation'].apply(lambda x: "+".join(sorted([a[1:-1] for a in x.split("+")])))
            dts_multiple.append(dts)

        dts = pd.concat(dts_multiple)
        return dts
if __name__ == "__main__":
    pass