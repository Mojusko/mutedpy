import numpy as np
from sklearn.metrics import f1_score as f1_sklearn
from sklearn.metrics import precision_score
import torch

def hit_rate(y_true,y_predicted, threshold = 1.2):
    args = np.argsort(y_predicted.reshape(-1)).numpy()[::-1]
    y_predicted = y_predicted.reshape(-1).numpy()[args]
    y_true = y_true.view(-1).numpy()[args]
    y_true_binary = y_true > threshold
    y_predicted_binary = y_predicted > threshold
    return precision_score(y_true_binary.astype(int), y_predicted_binary.astype(int))

def f1_score(y_true,y_predicted, threshold = 1.2):
    args = np.flipud(np.argsort(y_predicted.view(-1)))
    y_predicted = y_predicted.view(-1).numpy()[args]
    y_true = y_true.view(-1).numpy()[args]
    y_true_binary = y_true > threshold
    y_predicted_binary = y_predicted > threshold
    return f1_sklearn(y_true_binary.astype(int), y_predicted_binary.astype(int), average='macro')

def enrichment_factor(y_true,y_predicted, threshold = 1.2, alpha = 0.02):
    args = np.flipud(np.argsort(y_predicted.view(-1)))
    y_predicted = y_predicted.view(-1).numpy()[args]
    y_true = y_true.view(-1).numpy()[args]
    y_true_binary = y_true > threshold
    y_predicted_binary = y_predicted > threshold
    A_alpha = np.sum(y_predicted_binary[0:int(y_predicted_binary.shape[0]*alpha)])
    N_active = np.sum(y_true_binary)
    return (A_alpha/N_active)/(alpha/100)

def enrichment_area(y_true,y_predicted,alpha = 0.02):
    args = np.flipud(np.argsort(y_predicted.view(-1)))
    y_predicted = y_predicted.view(-1).numpy()[args]

    args = np.flipud(np.argsort(y_true.view(-1)))
    y_true = y_true.view(-1).numpy()[args]

    A_alpha = np.mean(y_predicted[0:int(y_predicted.shape[0] * alpha)])
    A_true = np.mean(y_true[0:int(y_true.shape[0] * alpha)])

    return (A_alpha / A_true)


if __name__ == "__main__":
    a = np.random.randn(10)
    b = np.random.randn(10)
    print (hit_rate(a,b))