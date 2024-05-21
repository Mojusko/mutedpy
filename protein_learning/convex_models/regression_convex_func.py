import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from mutedpy.protein_learning.embeddings.amino_acid_embedding import AminoAcidEmbedding
from stpy.kernels import KernelFunction
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.embeddings.polynomial_embedding import CustomEmbedding
from stpy.embeddings.onehot_embedding import OnehotEmbedding
from pymanopt.manifolds import Euclidean
from mutedpy.protein_learning.regression.regression_basis import ProteinKernelLearner
from torchmin import minimize

from mutedpy.protein_learning.regression.regression_ards import ARDModelLearner


class ConvexModelLearner(ARDModelLearner):

    def __str__(self):
        return "convex_model"

    def fit(self, optimize=True, save_loc=None):
        self.estimate_noise_std()

        self.initial_scale = 1.

        self.load()
        phi_train = self.Embedding.embed(self.x_train)
        y_train = self.y_train
        self.phi_train = phi_train
        d = self.Embedding.m
        print("Feature size:", d)
        print ("Data size:", phi_train.size())
        self.reg_param = 10.

        def reg(gamma):
            return 0.5*(torch.sum(gamma**2))*self.reg_param
        def total_loss(gamma):

            projected_phi = phi_train@torch.diag(gamma / self.initial_scale)
            distances = torch.cdist(projected_phi,projected_phi, p=2)**2
            weights_stacked = torch.exp(-distances)
            thetas =[]

            for i in range(phi_train.size()[0]):
                w = weights_stacked[i, :]
                design_matrix = torch.einsum('ij,i,ik->jk', phi_train, w, phi_train)
                theta = torch.linalg.solve(design_matrix + (self.s**2)*torch.eye(d), phi_train.T @ (w.view(-1,1)* y_train))
                thetas.append(theta)
            self.thetas_stacked = torch.hstack(thetas)


            predictions_stacked = phi_train @ self.thetas_stacked
            new_loss = torch.sum((predictions_stacked - torch.tile(y_train, (1,y_train.size()[0])))**2 * weights_stacked)
            dist = torch.cdist(predictions_stacked, predictions_stacked, p=2)**2 * weights_stacked * weights_stacked.T
            new_loss += torch.sum(dist)
            new_loss = new_loss/2.
            loss = new_loss
            return loss + reg(gamma)

        # optimize gamma
        gamma = torch.ones(d, requires_grad=True).double()*1.

        if optimize:
            initial_loss = total_loss(gamma)
            print("Initial loss:", initial_loss)
            result = minimize(total_loss, gamma, method='bfgs', disp=2, max_iter = 5)
            print ("Optimization result:", result.x)
            self.gamma = result.x
        else:
            self.gamma = gamma

        projected_phi = phi_train @ torch.diag(self.gamma / self.initial_scale)
        distances = torch.cdist(projected_phi, projected_phi, p=2) ** 2
        weights_stacked = torch.exp(-distances)
        thetas = []


        for i in range(phi_train.size()[0]):
            w = weights_stacked[i, :]
            design_matrix = torch.einsum('ij,i,ik->jk', phi_train, w, phi_train)
            theta = torch.linalg.solve(design_matrix + (self.s**2)*torch.eye(d), phi_train.T @ (w.view(-1, 1) * y_train))
            thetas.append(theta)
        self.thetas_stacked = torch.hstack(thetas)
        self.fitted = True
        # final gamma
    def predict(self, x=None):

        if x is None:
            x = self.x_test
        else:
            pass
        phi_train = self.Embedding.embed(self.x_train)
        y_train = self.y_train
        phi_test = self.Embedding.embed(x)
        projected_phi = phi_train @ torch.diag(self.gamma / self.initial_scale)
        projected_test_phi = phi_test @ torch.diag(self.gamma / self.initial_scale)
        distances = torch.cdist(projected_phi, projected_test_phi, p=2) ** 2
        weights_stacked = torch.exp(-distances)

        plt.imshow(weights_stacked.detach().numpy())
        plt.colorbar()
        plt.show()

        predictions_stacked = (phi_test @ self.thetas_stacked)
        weights_stacked = weights_stacked/torch.sum(weights_stacked, dim = 0 )
        predictions_stacked = predictions_stacked*weights_stacked.T

        self.mu = torch.sum(predictions_stacked, dim = 1).detach()
        self.std = self.mu*0
        self.mu_train = y_train*0
        self.std_train =  y_train*0

        return self.mu, self.std
