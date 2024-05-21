import torch
from metricpy.aminoacid_embedding import ContactMap
import matplotlib.pyplot as plt
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from mutedpy.utils.structure_utils import sasa, number_of_hydrogen_bonds, distance_map_centre_mass

parent_struct_file = '../../data/streptavidin/2rtg.pdb'
file = PDBFile.read(parent_struct_file)
structure = file.get_structure(include_bonds = True, model=1)
structure = structure[structure.chain_id == "B"]
structure = structure[struc.filter_amino_acids(structure)]
receptor = structure[structure.hetero == False]

distance_map_centre_mass(receptor)


# x = torch.Tensor([[16, 10, 2, 0, 7], [18, 16, 2, 0, 13]])
# emb = ContactMap("features.csv")
# emb.restrict_to_varaint()
# embedding = emb.embed(emb.x)
#
# plt.imshow(embedding)
# plt.show()