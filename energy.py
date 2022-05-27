# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause, University of Oviedo

import numpy as np

from sklearn.metrics.pairwise import manhattan_distances

from base import UsingClassifiers
from optimization import compute_ed_param_train, compute_ed_param_test, solve_ed


class EDy(UsingClassifiers):

    def __init__(self, estimator_train=None, estimator_test=None, distance=manhattan_distances, verbose=0):
        super(EDy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                  needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # variables to represent the distributions using the main idea of ED-based algorithms
        self.distance = distance
        self.train_n_cls_i_ = None
        self.train_distrib_ = None
        self.K_ = None
        #  variables for solving the optimization problem
        self.G_ = None
        self.C_ = None
        self.b_ = None
        self.a_ = None

    def fit(self, X, y, predictions_train=None):

        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Computing average distances for training distribution...' % self.__class__.__name__,
                  end='')

        n_classes = len(self.classes_)

        self.train_distrib_ = dict.fromkeys(self.classes_)
        self.train_n_cls_i_ = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            self.train_distrib_[cls] = self.predictions_train_[self.y_ext_ == cls]
            self.train_n_cls_i_[n_cls, 0] = len(self.train_distrib_[cls])

        self.K_, self.G_, self.C_, self.b_ = compute_ed_param_train(self.distance, self.train_distrib_,
                                                                    self.classes_, self.train_n_cls_i_)
        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        self.a_ = compute_ed_param_test(self.distance, self.train_distrib_, self.predictions_test_, self.K_,
                                        self.classes_, self.train_n_cls_i_)

        prevalences = solve_ed(G=self.G_, a=self.a_, C=self.C_, b=self.b_)

        if self.verbose > 0:
            print('done')

        return prevalences
