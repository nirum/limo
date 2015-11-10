import numpy as np
from .utils import Feature
from scipy.stats import zscore
from pyret.filtertools import rolling_window

__all__ = ['Convolutional']


class Convolutional(Feature):
    def __init__(self, name, stimulus, history=1, dtype='float', zscore=(0., 1.)):
        """
        Parameters
        ----------
        feature: array_like (space, space, time)
        einsum: string
        """
        ndim = len(stimulus.shape)
        assert ndim <= 6, "Too many dimensions!"

        super().__init__(name)
        self.stimulus = rolling_window(np.array(stimulus), history, time_axis=0)
        self.ndim = self.stimulus.ndim
        self.dtype = dtype

        # get mean and std. dev. of the stimulus (passed in by user)
        self.mu = zscore[0]
        self.sigma = zscore[1]

        letters = 'tijklmn'
        self.einsum_proj = letters[:self.ndim] + ',' + \
            letters[1:self.ndim] + '->' + letters[0]

        self.einsum_avg = letters[:self.ndim] + ',' + \
            letters[0] + '->' + letters[1:self.ndim]

    def zscore(self, x):
        return (x - self.mu) / self.sigma

    def __getitem__(self, inds):
        return self.zscore(self.stimulus[inds].astype(self.dtype))

    def __call__(self, theta, inds=Ellipsis):
        return np.einsum(self.einsum_proj, self[inds], theta.astype(self.dtype))

    def weighted_average(self, weights, inds=Ellipsis):

        try:
            L = float(len(inds))
        except TypeError:
            L = float(self.stimulus.shape[0])

        return np.einsum(self.einsum_avg, self[inds], weights.astype(self.dtype)) / L

    @property
    def shape(self):
        return self.stimulus.shape[1:]

    def clip(self, length):
        """Clips this feature"""
        self.stimulus = self.stimulus[-length:, ...]

    def __len__(self):
        return self.stimulus.shape[0]
