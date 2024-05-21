from mutedpy.utils.sequences.sequence_utils import remove_unspecific_mutations
import pandas as pd

dts = pd.DataFrame(["A21B","B44*"])
dts.columns = ["Mutation"]
print (dts)
print (remove_unspecific_mutations(dts))