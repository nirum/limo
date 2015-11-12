import numpy as np
from .utils import batchify, epochify
from scipy.stats import zscore
from pyret.filtertools import rolling_window
from .algorithms import adam

__all__ = ['Feature']


class Feature:

    def __init__(self, stimulus, theta_init, history=1, dtype='float', zscore=(0., 1.)):

        ndim = len(stimulus.shape)
        assert ndim <= 6, "Too many dimensions!"

        self.stimulus = rolling_window(np.array(stimulus), history, time_axis=0)
        self.ndim = self.stimulus.ndim
        self.dtype = dtype

        # get mean and std. dev. of the stimulus (passed in by user)
        self.mu = zscore[0]
        self.sigma = zscore[1]

        self.optimizer = adam(theta_init.astype(dtype), learning_rate=1e-2)
        self.theta = self.optimizer.send(None)

        letters = 'tijklmn'
        self.einsum_proj = letters[:self.ndim] + ',' + \
            letters[1:self.ndim] + '->' + letters[0]

        self.einsum_avg = letters[:self.ndim] + ',' + \
            letters[0] + '->' + letters[1:self.ndim]

    def zscore(self, x):
        return (x - self.mu) / self.sigma

    def __getitem__(self, inds):
        self.minibatch = self.zscore(self.stimulus[inds].astype(self.dtype))
        return np.einsum(self.einsum_proj, self.minibatch, self.theta)

    def __call__(self, err):
        gradient = np.einsum(self.einsum_avg, self.minibatch, err) / float(err.size)
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
