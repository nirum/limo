import numpy as np

__all__ = ['Stimulus', 'Feature', 'ComparableMixin']


class ComparableMixin(object):
    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            # _cmpkey not implemented, or return different type,
            # so I can't compare with "other".
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)


class Feature(ComparableMixin):

    def __init__(self, name):
        self.name = name

    def __call__(self, theta):
        raise NotImplementedError

    def weighted_average(r):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return 'Feature: ' + self.name

    def _cmpkey(self):
        return self.name

    def __len__(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError


class Stimulus(Feature):
    def __init__(self, name, stim, dtype='float'):
        """
        Parameters
        ----------
        feature: array_like (space, space, time)
        einsum: string
        """
        assert stim.ndim <= 6, "Too many dimensions!"

        super().__init__(name)
        self.feature = stim
        self.ndim = stim.ndim
        self.dtype = dtype

        letters = 'tijklmn'
        self.einsum_proj = letters[:self.ndim] + ',' + \
            letters[1:self.ndim] + '->' + letters[0]

        self.einsum_avg = letters[:self.ndim] + ',' + \
            letters[0] + '->' + letters[1:self.ndim]

    def __call__(self, theta, inds=None):
        if inds is None:
            return np.einsum(self.einsum_proj, self.feature.astype(self.dtype), theta)
        else:
            return np.einsum(self.einsum_proj, self.feature[inds, ...].astype(self.dtype), theta)

    def weighted_average(self, weights, inds=None):
        if inds is None:
            return np.einsum(self.einsum_avg, self.feature.astype(self.dtype), weights) \
                / float(weights.size)
        else:
            return np.einsum(self.einsum_avg, self.feature[inds, ...].astype(self.dtype), weights) \
                / float(len(inds))

    @property
    def shape(self):
        return self.feature.shape[1:]

    def clip(self, length):
        """Clips this feature"""
        self.feature = self.feature[..., -length:]

    def __len__(self):
        return self.feature.shape[0]
