# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause, University of Oviedo

import math
import numpy as np

from scipy.stats import norm

from base import UsingClassifiers
from search import global_search, mixture_of_pdfs
from optimization import solve_hd, compute_l2_param_train, solve_l1, solve_l2


class DFy(UsingClassifiers):

    def __init__(self, estimator_train=None, estimator_test=None, distribution_function='PDF', n_bins=8,
                 bin_strategy='equal_width', distance='HD', tol=1e-05, verbose=0):
        super(DFy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                  needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # attributes
        self.distribution_function = distribution_function
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.distance = distance
        self.tol = tol
        #  variables to compute the histograms
        self.bincuts_ = None
        # variables to represent the distributions
        self.classes_ = None
        self.train_distrib_ = None
        self.test_distrib_ = None
        # variables for solving the optimization problem
        self.G_ = None
        self.C_ = None
        self.b_ = None
        self.problem_ = None
        self.mixtures_ = None

    def fit(self, X, y, predictions_train=None):
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating training distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        if n_classes == 2:
            n_descriptors = 1  # number of groups of probabilities used to represent the distribution
        else:
            n_descriptors = n_classes

        # compute bincuts according to bin_strategy
        self.bincuts_ = np.zeros((n_descriptors, self.n_bins + 1))
        for descr in range(n_descriptors):
            self.bincuts_[descr, :] = compute_bincuts(x=self.predictions_train_[:, descr], y=y, classes=self.classes_,
                                                      n_bins=self.n_bins, bin_strategy=self.bin_strategy,
                                                      att_range=[0, 1])

        # compute pdf
        self.train_distrib_ = np.zeros((self.n_bins * n_descriptors, n_classes))
        for n_cls, cls in enumerate(self.classes_):
            for descr in range(n_descriptors):
                self.train_distrib_[descr * self.n_bins:(descr + 1) * self.n_bins, n_cls] = \
                   np.histogram(self.predictions_train_[self.y_ext_ == cls, descr], bins=self.bincuts_[descr, :])[0]
            self.train_distrib_[:, n_cls] = self.train_distrib_[:, n_cls] / (np.sum(self.y_ext_ == cls))

        # compute cdf if necessary
        if self.distribution_function == 'CDF':
            self.train_distrib_ = np.cumsum(self.train_distrib_, axis=0)

        if self.distance == 'L2':
            self.G_, self.C_, self.b_ = compute_l2_param_train(self.train_distrib_, self.classes_)

        if self.verbose > 0:
            print('done')

        self.problem_ = None
        self.mixtures_ = None

        return self

    def predict(self, X, predictions_test=None):
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Estimating testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        if n_classes == 2:
            n_descriptors = 1
        else:
            n_descriptors = n_classes

        self.test_distrib_ = np.zeros((self.n_bins * n_descriptors, 1))
        # compute pdf
        for descr in range(n_descriptors):
            self.test_distrib_[descr * self.n_bins:(descr + 1) * self.n_bins, 0] = \
                np.histogram(self.predictions_test_[:, descr], bins=self.bincuts_[descr, :])[0]

        self.test_distrib_ = self.test_distrib_ / len(self.predictions_test_)

        #  compute cdf if necessary
        if self.distribution_function == 'CDF':
            self.test_distrib_ = np.cumsum(self.test_distrib_, axis=0)

        if self.verbose > 0:
            print('Class %s: Computing prevalences...' % self.__class__.__name__, end='')

        if self.distance == 'HD':
            self.problem_, prevalences = solve_hd(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                                  n_classes=n_classes, problem=self.problem_)
        elif self.distance == 'L2':
            prevalences = solve_l2(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   G=self.G_, C=self.C_, b=self.b_)
        elif self.distance == 'L1':
            self.problem_, prevalences = solve_l1(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                                  n_classes=n_classes, problem=self.problem_)
        else:
            self.mixtures_, prevalences = global_search(distance_func=self.distance, mixture_func=mixture_of_pdfs,
                                                        test_distrib=self.test_distrib_, tol=self.tol,
                                                        mixtures=self.mixtures_, return_mixtures=True,
                                                        pos_distrib=self.train_distrib_[:, 1].reshape(-1, 1),
                                                        neg_distrib=self.train_distrib_[:, 0].reshape(-1, 1))

        if self.verbose > 0:
            print('done')

        return prevalences


class HDy(DFy):
    def __init__(self, estimator_train=None, estimator_test=None, n_bins=8, bin_strategy='equal_width', tol=1e-05, verbose=0):
        super(HDy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                  distribution_function='PDF', n_bins=n_bins, bin_strategy=bin_strategy,
                                  distance='HD', tol=tol, verbose=verbose)


############
# Function to compute histograms
############
def compute_bincuts(x, y=None, classes=None, n_bins=8, bin_strategy='equal_width', att_range=None):
    if bin_strategy == 'equal_width':
        bincuts = np.zeros(n_bins + 1)
        if att_range is None:
            att_range = [x.min(), x.max()]
        bincuts[0] = -np.inf
        bincuts[-1] = np.inf
        bincuts[1:-1] = np.histogram_bin_edges(x, bins=n_bins, range=att_range)[1:-1]
    elif bin_strategy == 'equal_count':
        sorted_values = np.sort(x)
        bincuts = np.zeros(n_bins + 1)
        bincuts[0] = -np.inf
        bincuts[-1] = np.inf
        for i in range(1, n_bins):
            cutpoint = int(round(len(x) * i / n_bins))
            bincuts[i] = (sorted_values[cutpoint - 1] + sorted_values[cutpoint]) / 2
    elif bin_strategy == 'binormal':
        # only for binary quantification
        n_classes = len(classes)
        if n_classes != 2:
            raise ValueError('binormal method can only be used for binary quantification')

        mu = 0
        std = 0
        for n_cls, cls in enumerate(classes):
            mu = mu + np.mean(x[y == cls])
            std = std + np.std(x[y == cls])
        mu = mu / n_classes
        std = std / n_classes
        if std > 0:
            bincuts = [std * norm.ppf(i / n_bins) + mu for i in range(0, n_bins + 1)]
        else:
            bincuts = np.histogram_bin_edges(x, bins=n_bins, range=[0, 0])
    elif bin_strategy == 'normal':
        weights = np.ones((x.shape[0],))
        for n_cls, cls in enumerate(classes):
            weights[y == cls] = x.shape[0] / np.sum(y == cls)

        mu = np.average(x, weights=weights)
        std = math.sqrt(np.average((x - mu) ** 2, weights=weights))
        if std > 0:
            bincuts = [std * norm.ppf(i / n_bins) + mu for i in range(0, n_bins + 1)]
        else:
            bincuts = np.histogram_bin_edges(x, bins=n_bins, range=[0, 0])
    else:
        raise ValueError('Unknown bin strategy (possible values: ''equal_width'', ''binormal'', ''normal''')

    return bincuts
