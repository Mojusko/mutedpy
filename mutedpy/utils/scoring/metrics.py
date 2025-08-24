import numpy as np 
import sklearn
import torch 

def classic_eval(mu, std, y_test):
	RMSD = torch.mean((mu - y_test)**2)
	coverage = torch.mean((y_test <= mu+2*std).double() * (y_test >= mu-2*std).double())
	r2 = sklearn.metrics.r2_score(y_test,mu)
	return RMSD, coverage, r2


import numpy as np
from scipy import stats
from sklearn import metrics as skm
from typing import Union, List


def accuracy_fn(y_true: Union[List[float], np.ndarray],
                y_pred: Union[List[float], np.ndarray]):
    """Calculates the accuracy score"""
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred).astype(int)
        y_true = np.asarray(y_true).astype(int)
    return np.mean(y_true == y_pred)


def pearsonr_fn(y_true: Union[List[float], np.ndarray],
                y_pred: Union[List[float], np.ndarray]):
    """Calculates the pearsonr correlation coefficient"""
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred).astype(float)
        y_true = np.asarray(y_true).astype(float)
    r, pval = stats.pearsonr(y_true, y_pred)
    return r


def rmse_fn(y_true: Union[List[float], np.ndarray],
                y_pred: Union[List[float], np.ndarray]):
    """Calculates the root mean squared error (RMSE)."""
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred).astype(float)
        y_true = np.asarray(y_true).astype(float)
    mse = skm.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def r2_fn(y_true: Union[List[float], np.ndarray],
                y_pred: Union[List[float], np.ndarray]):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred).astype(float)
        y_true = np.asarray(y_true).astype(float)
    r2 = skm.r2_score(y_true, y_pred)
    return r2


def spearmanr_fn(y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]):
    """Calculates the spearmanr correlation coefficient"""
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred).astype(float)
        y_true = np.asarray(y_true).astype(float)
    return stats.spearmanr(y_true, y_pred)[0]


def f1_fn(y_true: Union[List[float], np.ndarray],
                y_pred: Union[List[float], np.ndarray]):
    """Calculates the F1 score for imbalanced classification tasks."""
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred).astype(int)
        y_true = np.asarray(y_true).astype(int)
    return skm.f1_score(y_true=y_true, y_pred=y_pred)


def precision_fn(y_true: Union[List[float], np.ndarray],
                y_pred: Union[List[float], np.ndarray]):
    """Calculates the precision score for imbalanced classification tasks."""
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred).astype(int)
        y_true = np.asarray(y_true).astype(int)
    return skm.precision_score(y_true=y_true, y_pred=y_pred)


def recall_fn(y_true: Union[List[float], np.ndarray],
                y_pred: Union[List[float], np.ndarray]):
    """Calculates the recall score for imbalanced classification tasks."""
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred).astype(int)
        y_true = np.asarray(y_true).astype(int)
    return skm.recall_score(y_true=y_true, y_pred=y_pred)


METRICS = {'accuracy': (accuracy_fn, 0.0, np.greater),
           'pearsonr': (pearsonr_fn, 0.0, np.greater),
           'loss': (None, np.inf, np.less),
           'rmse': (rmse_fn, np.inf, np.less),
           'f1': (f1_fn, 0.0, np.greater),
           'precision': (precision_fn, 0.0, np.greater),
           'recall':(recall_fn, 0.0, np.greater),
           'spearmanr': (spearmanr_fn, 0.0, np.greater),
           'r2': (r2_fn, 0.0, np.greater)}

DATASET_METRICS = ['rmse', 'pearsonr', 'spearmanr', 'r2']

EVAL_METRICS = 'rmse'
