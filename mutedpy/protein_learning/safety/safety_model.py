import numpy as np
import torch
from mutedpy.utils.protein_operator import ProteinOperator
from mutedpy.experiments.streptavidin.active_learning.calculate_predictions import prediction_step_by_step
class SafetyModel():

    def __init__(self):
        pass

    def predict(self):
        pass

    def set_safety_limit(self):
        pass

    def set_safety_type(self):
        pass

    def query_safe(self, xtest):
        pass

class NeuralMutationSafetyModel(SafetyModel):

    def __init__(self, positions = [6]):
        self.positions = positions

    def query_safe(self, list):
        final_unsafe_mask = torch.zeros(size = (len(list),1)).view(-1).bool()
        for i,seq in enumerate(list):
            muts = seq.split("+")
            for p in self.positions:
                if muts[p][0] == muts[p][-1]:
                    final_unsafe_mask[i] = 1
        return ~final_unsafe_mask

class ResidueBlockSafetyModel():

    def __init__(self, unsafe_key_value_pair):
        self.Op = ProteinOperator()
        self.unsafe_key_value_pair = unsafe_key_value_pair
        self.unsafe = {}
        for key in unsafe_key_value_pair.keys():
            self.unsafe[key] = []
            for aa in unsafe_key_value_pair[key]:
                self.unsafe[key].append(self.Op.dictionary[aa])
                #print (key,self.Op.dictionary[aa])

    def query_safe(self, xtest):
        final_unsafe_mask = None
        for key in self.unsafe:
            for aa in self.unsafe[key]:
                mask = xtest[:,key] == aa
                if final_unsafe_mask is None:
                    final_unsafe_mask = mask
                else:
                    final_unsafe_mask = torch.logical_or(mask,final_unsafe_mask)
        return ~final_unsafe_mask

class GPSafetyModel():

    def __init__(self, GP,embed, safety_limit = (2+np.log(0.05))/2, safety_type = 'lcb'):
        """

        :param GP:
        :param safety_limit:
        :param safety_type:
        """
        self.safety_limit = safety_limit
        self.safety_type  = safety_type
        self.GP = GP
        self.embed = embed
        self.mem_limit = 32000
    def query_safe(self, xtest):
        vals = self.evaluate(xtest)
        safe = vals > self.safety_limit
        return safe

    def evaluate(self,xtest):
        phitest = self.embed(xtest)
        if xtest.size()[0]>self.mem_limit:
            mu, std = prediction_step_by_step(self.GP,phitest,self.mem_limit)
        else:
            mu, std = self.GP.mean_std(phitest)
        if self.safety_type == 'lcb':
            return mu - 2*std
        elif self.safety_type == 'ucb':
            return mu + 2*std

if __name__ == "__main__":
    from mutedpy.experiments.streptavidin.streptavidin_loader import load_first_round
    from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model
    x,y,dts = load_first_round()

    unsafe = {0:['W'], 1:['C','W'], 2:['W'], 3:['W'], 4:['W']}

    safety_model = ResidueBlockSafetyModel(unsafe)
    safety_model.query_safe(x)

    params = "../../experiments/streptavidin/active_learning_2/OD_model/params/final_model_params.p"
    GP, embed = load_model(params,"od model")

    safety_model_od = GPSafetyModel(GP,embed)
    print (safety_model_od.query_safe(x))

