"""
Base classes for all quantifiers
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause, University of Oviedo

import numpy as np
import six
from abc import ABCMeta

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_X_y, check_array


class BaseQuantifier(six.with_metaclass(ABCMeta, BaseEstimator)):
    pass


class UsingClassifiers(BaseQuantifier):

    def __init__(self, estimator_train=None, estimator_test=None, needs_predictions_train=True,
                 probabilistic_predictions=True, verbose=0, **kwargs):
        super(UsingClassifiers, self).__init__(**kwargs)
        # init attributes
        self.estimator_train = estimator_train
        self.estimator_test = estimator_test
        self.needs_predictions_train = needs_predictions_train
        self.probabilistic_predictions = probabilistic_predictions
        self.verbose = verbose
        # computed attributes
        self.predictions_test_ = None
        self.predictions_train_ = None
        self.classes_ = None
        self.y_ext_ = None

    def fit(self, X, y, predictions_train=None):
        self.classes_ = np.unique(y)

        if self.needs_predictions_train and self.estimator_train is None and predictions_train is None:
            raise ValueError("estimator_train or predictions_train must be not None "
                             "with objects of class %s", self.__class__.__name__)

        # Fit estimators if they are not already fitted
        if self.estimator_train is not None:
            if self.verbose > 0:
                print('Class %s: Fitting estimator for training distribution...' % self.__class__.__name__, end='')
            # we need to fit the estimator for the training distribution
            # we check if the estimator is trained or not
            try:
                self.estimator_train.predict(X[0:1, :].reshape(1, -1))
                if self.verbose > 0:
                    print('it was already fitted')

            except NotFittedError:

                X, y = check_X_y(X, y, accept_sparse=True)

                self.estimator_train.fit(X, y)

                if self.verbose > 0:
                    print('fitted')

        if self.estimator_test is not None:
            if self.verbose > 0:
                print('Class %s: Fitting estimator for testing distribution...' % self.__class__.__name__, end='')

            # we need to fit the estimator for the testing distribution
            # we check if the estimator is trained or not
            try:
                self.estimator_test.predict(X[0:1, :].reshape(1, -1))
                if self.verbose > 0:
                    print('it was already fitted')

            except NotFittedError:

                X, y = check_X_y(X, y, accept_sparse=True)

                self.estimator_test.fit(X, y)

                if self.verbose > 0:
                    print('fitted')

        # Compute predictions_train_
        if self.verbose > 0:
            print('Class %s: Computing predictions for training distribution...' % self.__class__.__name__, end='')

        if self.needs_predictions_train:
            if predictions_train is not None:
                if self.probabilistic_predictions:
                    self.predictions_train_ = predictions_train
                else:
                    self.predictions_train_ = UsingClassifiers.__probs2crisps(predictions_train, self.classes_)
            else:
                if self.probabilistic_predictions:
                    self.predictions_train_ = self.estimator_train.predict_proba(X)
                else:
                    self.predictions_train_ = self.estimator_train.predict(X)

            # Compute y_ext_
            if len(y) == len(self.predictions_train_):
                self.y_ext_ = y
            else:
                self.y_ext_ = np.tile(y, len(self.predictions_train_) // len(y))

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        if self.estimator_test is None and predictions_test is None:
            raise ValueError("estimator_test or predictions_test must be not None "
                             "to compute a prediction with objects of class %s", self.__class__.__name__)

        if self.verbose > 0:
            print('Class %s: Computing predictions for testing distribution...' % self.__class__.__name__, end='')

        # At least one between estimator_test and predictions_test is not None
        if predictions_test is not None:
            if self.probabilistic_predictions:
                self.predictions_test_ = predictions_test
            else:
                self.predictions_test_ = UsingClassifiers.__probs2crisps(predictions_test, self.classes_)
        else:
            check_array(X, accept_sparse=True)
            if self.probabilistic_predictions:
                self.predictions_test_ = self.estimator_test.predict_proba(X)
            else:
                self.predictions_test_ = self.estimator_test.predict(X)

        if self.verbose > 0:
            print('done')

        return self

    @staticmethod
    def __probs2crisps(preds, labels):
        if len(preds) == 0:
            return preds
        if preds.ndim == 1 or preds.shape[1] == 1:
            #  binary problem
            if preds.ndim == 1:
                preds_mod = np.copy(preds)
            else:
                preds_mod = np.copy(preds.squeeze())
            if isinstance(preds_mod[0], np.float):
                # it contains probs
                preds_mod[preds_mod >= 0.5] = 1
                preds_mod[preds_mod < 0.5] = 0
                return preds_mod.astype(int)
            else:
                return preds_mod
        else:
            # multiclass problem
            return labels.take(preds.argmax(axis=1), axis=0)
