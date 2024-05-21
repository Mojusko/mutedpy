import torch
from sklearn.linear_model import LassoCV, HuberRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.linear_model import ElasticNetCV

class FeatureSelector():

    def __init__(self, njobs=None):
        self.njobs = njobs

    def pass_data(self, x, y):
        self.x = x
        self.y = y
        print ("Dada feature size size", self.x.size()[1])

    def select(self, topk):
        pass


class DummyFeatureSelector(FeatureSelector):

    def select(self, topk):
        return torch.arange(0,self.x.size()[1], 1)

class RandomFeatureSelector(FeatureSelector):

    def select(self, topk):
        return torch.from_numpy(np.random.choice(torch.arange(0,self.x.size()[1], 1).numpy(),size = topk))

class ElasticNetFeatureSelector(FeatureSelector):

    def select(self, topk):
        ratios = [.1, .5, .7, .9, .95, .99, 1]
        print ("Selecting features via LASSO and cross validation.")
        print ("Number of jobs:", self.njobs)
        regr = ElasticNetCV(l1_ratio=ratios, cv=10,
                       n_alphas=100,
                       random_state=1,
                       max_iter=1000,
                       n_jobs=self.njobs,
                       verbose=1
                       )
        regr.fit(self.x.numpy(), self.y.numpy().ravel())
        print ("Feature selection finished...")
        if topk == "all":
            d = torch.from_numpy(regr.coef_).size()[0]
            print("All features correspond to:", d)
        else:
            if torch.sum(torch.from_numpy(regr.coef_) > 0.)<topk:
                d = torch.sum(torch.from_numpy(regr.coef_) > 0.)
            else:
                d = topk
        self.feature_mask = torch.topk(torch.abs(torch.from_numpy(regr.coef_)), k=d)[1]
        return self.feature_mask

class LassoFeatureSelector(FeatureSelector):

    def select(self, topk):
        print ("Selecting features via LASSO and cross validation.")
        print ("Number of jobs:", self.njobs)
        regr = LassoCV(cv=10,
                       n_alphas=100,
                       random_state=1,
                       max_iter=1000,
                       n_jobs=self.njobs,
                       verbose=1
                       )
        regr.fit(self.x.numpy(), self.y.numpy().ravel())

        print ("Feature selection finished...")
        if topk == "all":
            d = torch.from_numpy(regr.coef_).size()[0]
            print("All features correspond to:", d)
        else:
            if torch.sum(torch.from_numpy(regr.coef_) > 0.)<topk:
                d = torch.sum(torch.from_numpy(regr.coef_) > 0.)
            else:
                d = topk
        self.feature_mask = torch.topk(torch.abs(torch.from_numpy(regr.coef_)), k=d)[1]
        return self.feature_mask


class RFFeatureSelector(FeatureSelector):


    def select(self, topk):
        print ("Selecting features via Random Forests and cross validation.")
        self.regr = RandomForestRegressor(max_features='auto', max_depth=15,
                                          random_state=0, min_samples_split=5, verbose = 3,
                                          n_estimators=500, n_jobs=self.njobs)

        self.regr.fit(self.x.numpy(), self.y.numpy().ravel())
        coef_ = self.regr.feature_importances_ / np.sum(self.regr.feature_importances_)
        self.feature_mask = torch.topk(torch.abs(torch.from_numpy(coef_)), k=topk)[1]
        return self.feature_mask
