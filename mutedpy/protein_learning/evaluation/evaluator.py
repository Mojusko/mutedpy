import copy
import torch
import numpy as np
import pandas as pd
import multiprocessing
import ray
from mutedpy.protein_learning.data_splits import load_splits

class Evaluator():

    def __init__(self,name,x,y,model,params, save_model = False):
        self.name = name
        self.model = model
        self.x = x
        self.y = y
        self.params = params
        self.save_model = save_model


    def load_splits(self, split_location, no_splits):
        return load_splits(no_splits,split_location)

    def get_splits(self, no_splits = 20, n_test = 150, split_location = None):

        if split_location is None:
            pass
            #dts = self.model_obj.cv_split_eval(splits=no_splits, n_test=n_test)
        else:
            #dts = self.model_obj.load_splits_eval(split_location, splits=no_splits)
            dts = self.load_splits(split_location, no_splits)
        return dts


    def save_results(self, names, datas, output_file = 'results_strep.txt', file = True):

        with open(self.params['results_folder'] + "/" + output_file, 'w') as f:

            print("%20s, %20s: %8s %8s %8s %8s %8s %8s" % (
            "model name", "quantity", "Q10", "Q25", "Q50", "Q75", "Q90", "std"))

            for name, data in zip(names, datas):
                print("%20s, %20s: %8f %8f %8f %8f %8f %8f" % (
                str(self.name ), name, torch.quantile(data, q=0.1), torch.quantile(data, q=0.25),
                torch.quantile(data, q=0.5), torch.quantile(data, q=0.75), torch.quantile(data, q=0.9),
                torch.std(data)))
                print("%20s, %20s: %8f %8f %8f %8f %8f %8f" % (
                str(self.name), name, torch.quantile(data, q=0.1), torch.quantile(data, q=0.25),
                torch.quantile(data, q=0.5), torch.quantile(data, q=0.75), torch.quantile(data, q=0.9),
                torch.std(data)), file=f)

        info = {}
        for name, data in zip(names, datas):
            info[name] = data
        dts = pd.DataFrame(info)
        dts.to_csv(self.params['results_folder'] + "/raw" + output_file[0:-4] + ".csv")

    def evaluate_metrics_on_splits(self, no_splits = 20,
                                   n_test = 150,
                                   split_location = None,
                                   output_file = 'output.txt',
                                    special_identifier = '',
                                   parallel = True
                                   ):

        save_model = self.save_model
        dts = self.get_splits(no_splits = no_splits, n_test = n_test, split_location = split_location)

        splits = no_splits
        self.splits = splits
        self.special_identifier = special_identifier

        rmsds = torch.zeros(size=(splits, 1)).view(-1).double()
        coverages = torch.zeros(size=(splits, 10)).double()
        coverages_cons = torch.zeros(size=(splits, 10)).double()
        r2s = torch.zeros(size=(splits, 1)).view(-1).double()
        r2sstd = torch.zeros(size=(splits, 1)).view(-1).double()
        pearson = torch.zeros(size=(splits, 1)).view(-1).double()
        spearman = torch.zeros(size=(splits, 1)).view(-1).double()
        hit_rates = torch.zeros(size=(splits, 1)).view(-1).double()
        enrichment_factors = torch.zeros(size=(splits, 1)).view(-1).double()
        enrichment_areas = torch.zeros(size=(splits, 1)).view(-1).double()
        f1_scores = torch.zeros(size=(splits, 1)).view(-1).double()

        alpha_range = np.arange(0.1, 1.1, 0.1)


        njobs = self.params['njobs']
        ncores = self.params['cores']
        @ray.remote(num_cpus=ncores)
        def evaluate(args):
            i, d, model_caller, x, y, params, splits, special_identifier, save_model = args
            model = model_caller(**params)
            model.add_data(x, y)
            s = model.evaluate_on_split(i, d, splits=splits, special_identifier=special_identifier,
                                        save_model=save_model)
            return s

        arguments = []
        for i, d in enumerate(dts):
            print ("Creating Argument", i)
            #feature_loader_params_list = [copy.deepcopy(x) for x in feature_loader_params_list]
            new_params = copy.copy(self.params)
            arguments.append((i,copy.copy(d),copy.copy(self.model),self.x.clone(),self.y.clone(),new_params, self.splits, self.special_identifier,save_model))

        # multi-job execution - multi-core
        if parallel:
            print("NUMBER OF PARALLEL JOBS", njobs)
            print("NUMBER OF PARALLEL CORES", ncores)
            ray.init(num_cpus=njobs*ncores, object_store_memory=int(1e10))
            results = ray.get([evaluate.remote(arg) for arg in arguments])
            ray.shutdown()

        # single-job execution - multi-core
        else:
            results = []
            for arg in arguments:
                results.append(evaluate(arg))

        for index, v in enumerate(results):
            rmsds[index] = v[0]
            r2s[index] = v[1]
            r2sstd[index] = v[2]
            pearson[index] = v[3]
            spearman[index] = v[4]
            hit_rates[index] = v[5]
            enrichment_factors[index] = v[6]
            enrichment_areas[index] = v[7]
            f1_scores[index] = v[8]
            coverages[index, :] = v[9]
            coverages_cons[index, :] = v[10]

        # output evaluation
        names = ["RMSD", "r2", "r2std", "pearson", "spearman", "hit_rate", "ef", "ea", "f1"] + [
            "coverage_" + str(np.round(alpha, 1)) for alpha in alpha_range] + [
                    "coverage_cons_" + str(np.round(alpha, 1)) for alpha in alpha_range]
        datas = [rmsds, r2s, r2sstd, pearson, spearman, hit_rates, enrichment_factors, enrichment_areas, f1_scores] + [
            coverages[:, j] for j in range(10)] + [coverages_cons[:, j] for j in range(10)]

        self.save_results(names, datas, output_file=output_file)