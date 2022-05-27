# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause, University of Oviedo

import numpy as np

from base import UsingClassifiers
from metrics import mean_absolute_error


class EM(UsingClassifiers):

    def __init__(self, estimator_train=None, estimator_test=None, verbose=0, epsilon=1e-4, max_iter=1000):
        super(EM, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                 needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        self.epsilon_ = epsilon
        self.max_iter_ = max_iter
        self.prevalences_train_ = None

    def fit(self, X, y, predictions_train=None):
        super().fit(X, y, predictions_train=predictions_train)

        n_classes = len(self.classes_)

        freq = np.zeros(n_classes)
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls] = np.equal(y, cls).sum()

        self.prevalences_train_ = freq / float(len(y))

        return self

    def predict(self, X, predictions_test=None):
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Estimating prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        iterations = 0
        prevalences = np.copy(self.prevalences_train_)
        prevalences_prev = np.ones(n_classes)

        while iterations < self.max_iter_ and (mean_absolute_error(prevalences, prevalences_prev) > self.epsilon_
                                               or iterations < 10):

            nonorm_posteriors = np.multiply(self.predictions_test_, np.divide(prevalences, self.prevalences_train_))

            posteriors = np.divide(nonorm_posteriors, nonorm_posteriors.sum(axis=1, keepdims=True))

            prevalences_prev = prevalences
            prevalences = posteriors.mean(0)

            iterations = iterations + 1

        if self.verbose > 0:
            if iterations < self.max_iter_:
                print('done')
            else:
                print('done but it might have not converged, max_iter reached')

        return prevalences
