import numpy as np
from .mixins import Feature
from pyret.filtertools import rolling_window

__all__ = ['Convolutional', 'Feature']


class Convolutional(Feature):
    def __init__(self, name, stimulus, history=1, dtype='float'):
        """
        Parameters
        ----------
        feature: array_like (space, space, time)
        einsum: string
        """
        assert stimulus.ndim <= 6, "Too many dimensions!"

        super().__init__(name)
        self.stimulus = rolling_window(stimulus, history, time_axis=0)
        self.ndim = self.stimulus.ndim
        self.dtype = dtype

        letters = 'tijklmn'
        self.einsum_proj = letters[:self.ndim] + ',' + \
            letters[1:self.ndim] + '->' + letters[0]

        self.einsum_avg = letters[:self.ndim] + ',' + \
            letters[0] + '->' + letters[1:self.ndim]

    def __call__(self, theta, inds=None):

        if inds is None:
            return np.einsum(self.einsum_proj, self.stimulus.astype(self.dtype), theta)

        else:
            return np.einsum(self.einsum_proj, self.stimulus[inds, ...].astype(self.dtype), theta)

    def weighted_average(self, weights, inds=None):
        if inds is None:
            return np.einsum(self.einsum_avg, self.stimulus.astype(self.dtype), weights) \
                / float(weights.size)
        else:
            return np.einsum(self.einsum_avg, self.stimulus[inds, ...].astype(self.dtype), weights) \
                / float(len(inds))

    @property
    def shape(self):
        return self.stimulus.shape[1:]

    def clip(self, length):
        """Clips this feature"""
        self.stimulus = self.stimulus[-length:, ...]

    def __len__(self):
        return self.stimulus.shape[0]
