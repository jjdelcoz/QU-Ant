# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause, University of Oviedo

import numpy as np
import scipy

from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels


def check_prevalences(p_true, p_pred):
    check_consistent_length(p_true, p_pred)
    p_true = check_array(p_true, ensure_2d=False)
    p_pred = check_array(p_pred, ensure_2d=False)

    if p_true.ndim == 1:
        p_true = p_true.reshape((-1, 1))

    if p_pred.ndim == 1:
        p_pred = p_pred.reshape((-1, 1))

    if p_true.shape[1] != p_pred.shape[1]:
        raise ValueError("p_true and p_pred have different length")

    return p_true, p_pred


def kld(p_true, p_pred, eps=1e-12):
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.sum(p_true * np.log2(p_true / (p_pred + eps)))


def mean_absolute_error(p_true, p_pred):
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.mean(np.abs(p_pred - p_true))


def l1(p_true, p_pred):
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.sum(np.abs(p_true - p_pred))


def mean_squared_error(p_true, p_pred):
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.mean((p_pred - p_true)**2)


def l2(p_true, p_pred):
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.sqrt(np.sum((p_true - p_pred) ** 2))


def hd(p_true, p_pred):
    p_true, p_pred = check_prevalences(p_true, p_pred)
    dist = np.sqrt(np.sum((np.sqrt(p_pred) - np.sqrt(p_true)) ** 2))
    return dist


def topsoe(p_true, p_pred, epsilon=1e-20):
    p_true, p_pred = check_prevalences(p_true, p_pred)
    dist = np.sum(p_true*np.log((2*p_true+epsilon)/(p_true+p_pred+epsilon)) +
                  p_pred*np.log((2*p_pred+epsilon)/(p_true+p_pred+epsilon)))
    return dist


def geometric_mean(y_true, y_pred, correction=0.0):
    labels = unique_labels(y_true, y_pred)
    n_labels = len(labels)

    le = LabelEncoder()
    le.fit(labels)
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)
    sorted_labels = le.classes_

    tp = y_true == y_pred
    tp_bins = y_true[tp]

    if len(tp_bins):
        tp_sum = np.bincount(tp_bins, weights=None, minlength=len(labels))
    else:
        true_sum = tp_sum = np.zeros(len(labels))

    if len(y_true):
        true_sum = np.bincount(y_true, weights=None, minlength=len(labels))

    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]

    mask = true_sum == 0.0
    true_sum[mask] = 1  # avoid infs/nans
    recall = tp_sum / true_sum
    recall[mask] = 0
    recall[recall == 0] = correction

    with np.errstate(divide="ignore", invalid="ignore"):
        g_mean = scipy.stats.gmean(recall)
    return g_mean
