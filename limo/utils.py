"""
Shared utilities

"""
import numpy as np

__all__ = ['batchify', 'Feature']


def batchify(X, batch_size, randomize=True):
    """
    Returns a generator that yields batches of samples from X

    """

    inds = np.arange(X.shape[0])

    if randomize:
        np.random.shuffle(inds)

    while len(inds) > 0:

        # get a batch
        yield X[inds[:batch_size]]

        # delete indices for this batch from the pool
        inds = np.delete(inds, slice(batch_size))


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
