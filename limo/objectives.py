from __future__ import print_function
from copy import deepcopy
import numpy as np
from toolz import isdistinct, map, reduce, valmap

# __all__ = ['Objective', 'Feature']


# class Feature:

    # def __init__(self, stim):

        # self.stim = np.random.randn(1000, 5, 3, 3)
        # self.theta = np.random.rand(5, 3, 3)
        # # self.rate = np.random.rand(1000)
        # self.dt = 1e-2

    # def __getitem__(self, inds):
        # return self.stim[inds]

    # def __call__(self, rate, inds):

        # u = np.tensordot(self[inds], theta, theta.ndim)

# class Objective:

    # # def __init__(self,
    # def __init__(self, features):

        # self.features = features
        # self.theta_init = {f.name: 0.01 * np.random.randn(*f.shape) for f in features}

        # # self.stim = np.random.randn(1000, 5, 3, 3)
        # # self.theta_init = np.random.rand(5, 3, 3)
        # # self.rate = np.random.rand(1000)
        # # self.dt = 1e-2

    # def __call__(self, theta):

        # u = [np.tensordot(stim[inds], theta[key], theta[key].ndim) for f.name in self.features]
        # inds = slice(50)
        # utot = np.sum(u, axis=0)

        # rhat = np.exp(utot) * self.dt
        # err = rhat - self.rate[inds] * utot
        # obj = np.mean(err)

        # # grad =
        # T = float(rhat.size)
        # # grad = {f.name: np.tensordot(rhat - self.rate[inds], self.stim[inds], 1) / T for f.name in self.features}
        # grad = np.tensordot(rhat - self.rate[inds], self.stim[inds], 1) / T

        # return obj, grad

# # class Objective(object):
    # # """
    # # Experiment agnostic
    # # """

    # # def __init__(self, features, rate, dt, minibatch=None, holdout=0.2):
        # # """
        # # Initialize a GLM Objective

        # # Parameters
        # # ----------
        # # features : list
            # # List of Feature objects

        # # rate : array_like
            # # 1-D array of firing rates

        # # dt : float
            # # time step

        # # minibatch : int, optional
            # # If None, then trains on all the data. Otherwise, trains on this many
            # # time indices (Default: None)

        # # holdout : float in [0,1], optional
            # # Fraction of the data to 'holdout' for testing

        # # """

        # # assert isdistinct(f.name for f in features), \
            # # "All features must be unique"

        # # assert rate.ndim == 1, "Rate must be an array (1-D)"
        # # assert holdout >= 0 and holdout <= 1, "Holdout fraction must be between 0 and 1"

        # # self.minibatch = minibatch

        # # # clip features and rates to have the same size
        # # self.nsamples = min(map(len, features))
        # # list(f.clip(self.nsamples) for f in features)
        # # self.rate = rate[-self.nsamples:]

        # # # train / test split
        # # test_num = int(np.round(holdout * self.nsamples))
        # # inds = np.arange(self.nsamples)
        # # np.random.shuffle(inds)
        # # self.test_indices = inds[:test_num]
        # # self.train_indices = inds[test_num:]

        # # print(u'\u279b Initializing... ', end='', flush=True)
        # # self.features = features
        # # # self.averages = {f.name: f.weighted_average(self.rate) for f in features}
        # # self.averages = {f.name: np.random.randn(*f.shape) for f in features}
        # # self.theta_init = valmap(lambda x: x * 0.01, deepcopy(self.averages))
        # # self.dt = dt
        # # print('Done.')

    # # def __call__(self, theta):

        # # if self.minibatch:
            # # inds = np.random.choice(self.train_indices, self.minibatch, replace=False)
        # # else:
            # # inds = self.train_indices

        # # return self._objective(theta, inds)

    # # def _objective(self, theta, inds):

        # # # model rate
        # # rhat = self.response(theta, inds)

        # # # inner product of average features with the parameters
        # # avg_innerprod = (np.dot(avg.ravel(), theta[name].ravel()) \
                         # # for name, avg in self.averages.items())

        # # # neg. log-likelihood
        # # obj = self.dt * np.mean(rhat) - reduce(np.add, avg_innerprod)

        # # # gradient
        # # grad = {f.name: self.dt * f.weighted_average(rhat, inds=inds) - self.averages[f.name] \
                # # for f in self.features}

        # # return obj, grad

    # # def predict(self, theta):
        # # """
        # # Returns
        # # -------
        # # rhat : array_like
            # # predicted rate

        # # robs : array_like
            # # the true firing rate for the held out data
        # # """

        # # rhat = self.response(theta, self.test_indices) * self.dt
        # # return rhat, self.rate[self.test_indices]

    # # def test(self, theta):
        # # """
        # # Compute the neg. log-likelihood objective on the test set
        # # """

        # # return self._objective(theta, self.test_indices)[0]

    # # def response(self, theta, inds):
        # # return np.exp(reduce(np.add, (f(theta[f.name], inds=inds) for f in self.features)))

    # # def __len__(self):
        # # return self.nsamples
