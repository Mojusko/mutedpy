import pandas as pd

class Loader():
    pass

    def __init__(self, filename):
        self.filename = filename

    def load(self):
        dts = pd.read_hdf(self.filename)
        return dts

    def merge(self,loader_objects):
        pass
