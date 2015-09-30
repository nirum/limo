from __future__ import print_function
from copy import deepcopy
import numpy as np
from toolz import isdistinct, map, reduce

__all__ = ['Objective']


class Objective(object):
    """
    Experiment agnostic
    """

    def __init__(self, features, rate, dt, minibatch=None, holdout=0.2):
        """
        Initialize a GLM Objective

        Parameters
        ----------
        features : list
            List of Feature objects

        rate : array_like
            1-D array of firing rates

        dt : float
            time step

        minibatch : int, optional
            If None, then trains on all the data. Otherwise, trains on this many
            time indices (Default: None)

        holdout : float in [0,1], optional
            Fraction of the data to 'holdout' for testing

        """

        assert isdistinct(f.name for f in features), \
            "All features must be unique"

        assert rate.ndim == 1, "Rate must be an array (1-D)"
        assert holdout >= 0 and holdout <= 1, "Holdout fraction must be between 0 and 1"

        self.minibatch = minibatch

        # train / test split
        test_frac = int(np.round(holdout * rate.size))
        inds = np.arange(rate.size)
        self.test_indices = list(np.random.choice(inds, size=test_frac, replace=False))
        self.train_indices = list(set(inds) - set(self.test_indices))

        # clip features and rates to have the same size
        self.nsamples = min(map(len, features))
        list(f.clip(self.nsamples) for f in features)
        self.rate = rate[-self.nsamples:]

        print(u'\u279B Initializing... ', end='', flush=True)
        self.features = features
        self.averages = {f.name: f.weighted_average(self.rate) for f in features}
        self.theta_init = deepcopy(self.averages)
        self.dt = dt
        print('Done.')

    def __call__(self, theta):

        if self.minibatch:
            inds = np.random.choice(self.train_indices, self.minibatch, replace=False)
        else:
            inds = None

        # compute projection
        proj = reduce(np.add, (f(theta[f.name], inds=inds) for f in self.features))

        # model rate
        rhat = np.exp(proj)

        # inner product of average features with the parameters
        avg_innerprod = (np.dot(avg.ravel(), theta[name].ravel()) \
                         for name, avg in self.averages.items())

        # neg. log-likelihood
        obj = self.dt * np.mean(rhat) - reduce(np.add, avg_innerprod)

        # gradient
        grad = {f.name: self.dt * f.weighted_average(rhat, inds=inds) - self.averages[f.name] \
                for f in self.features}

        return obj, grad

    def predict(self, theta):
        """
        Returns
        -------
        rhat : array_like
            predicted rate

        robs : array_like
            the true firing rate for the held out data
        """

        proj = reduce(np.add, (f(theta[f.name], inds=self.test_indices) for f in self.features))
        rhat = np.exp(proj)
        return rhat, self.rate[self.test_indices]

    def test(self, theta):
        """
        Compute the neg. log-likelihood objective on the test set
        """

        # compute projection
        proj = reduce(np.add, (f(theta[f.name], inds=self.test_indices) for f in self.features))

        # model rate
        rhat = np.exp(proj)

        # inner product of average features with the parameters
        avg_innerprod = (np.dot(avg.ravel(), theta[name].ravel()) \
                         for name, avg in self.averages.items())

        # neg. log-likelihood
        obj = self.dt * np.mean(rhat) - reduce(np.add, avg_innerprod)

        # gradient
        grad = {f.name: self.dt * f.weighted_average(rhat, inds=self.test_indices) - self.averages[f.name] \
                for f in self.features}

        return obj, grad

    def __len__(self):
        return self.nsamples
