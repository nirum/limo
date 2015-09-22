from __future__ import print_function
from copy import deepcopy
import numpy as np
from toolz import isdistinct, map, reduce

__all__ = ['Objective']


class Objective(object):
    """
    Experiment agnostic
    """

    def __init__(self, features, rate, dt, minibatch=None):

        assert isdistinct(f.name for f in features), \
            "All features must be unique"

        assert rate.ndim == 1, "Rate must be an array (1-D)"

        self.minibatch = minibatch

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
            inds = np.random.choice(np.arange(self.rate.size), self.minibatch, replace=False)
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

    def __len__(self):
        return self.nsamples
