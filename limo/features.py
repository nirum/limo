import numpy as np
from .utils import batchify, epochify
from scipy.stats import zscore
from pyret.filtertools import rolling_window
from .algorithms import adam

__all__ = ['Feature']


class Feature:

    def __init__(self, stimulus, theta_init, lr, l2=1e-3, active=True):

        ndim = len(stimulus.shape)
        assert ndim <= 6, "Too many dimensions!"

        # get mean and std. dev. of the stimulus (passed in by user)
        # self.mu = zscore[0]
        # self.sigma = zscore[1]

        # self.stimulus = rolling_window(self.zscore(np.array(stimulus).astype(dtype)), history, time_axis=0)
        self.stimulus = stimulus
        self.ndim = self.stimulus.ndim
        # self.dtype = dtype
        self.l2 = l2

        self.active = active

        self.optimizer = adam(theta_init, learning_rate=lr)
        self.theta = self.optimizer.send(None)

        letters = 'tijklmn'
        self.einsum_proj = letters[:self.ndim] + ',' + \
            letters[1:self.ndim] + '->' + letters[0]

        self.einsum_avg = letters[:self.ndim] + ',' + \
            letters[0] + '->' + letters[1:self.ndim]

    def __getitem__(self, inds):
        self.minibatch = self.stimulus[inds]
        return np.einsum(self.einsum_proj, self.minibatch, self.theta)

    def __call__(self, err):
        gradient = np.einsum(self.einsum_avg, self.minibatch, err) / float(err.size)
        gradient += self.l2 * self.theta
        self.minibatch = None

        if self.active:
            self.theta = self.optimizer.send(gradient)

        return gradient

    @property
    def shape(self):
        return self.theta.shape

    def clip(self, length):
        """Clips this feature"""
        self.stimulus = self.stimulus[-length:, ...]

    def __len__(self):
        return self.stimulus.shape[0]
