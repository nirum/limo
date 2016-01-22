"""
Shared utilities

"""
import numpy as np
from pyret.filtertools import rolling_window

__all__ = ['batchify', 'holdout', 'preprocess', 'inner']


def holdout(batches, frac=0.1):
    """
    Take a list and split it into train and test sets

    """

    batches = list(batches)
    num_holdout = int(np.round(len(batches) * frac))

    test = batches[:num_holdout]
    train = batches[num_holdout:]

    return train, test


def batchify(t, n, randomize=True):
    """
    Take a length and break it up into batches

    """

    inds = np.arange(t)
    if randomize:
        np.random.shuffle(inds)

    while len(inds) > 0:

        yield inds[:n]
        inds = np.delete(inds, slice(n))


def preprocess(stimulus, history, zscore=(0., 1.)):
    """
    Preprocess stimulus array

    """

    stim = np.array(stimulus).astype('float')
    stim -= zscore[0]
    stim /= zscore[1]
    return rolling_window(stim, history, time_axis=0)


def inner(x, y):
    """
    Inner product between arrays

    """

    return np.inner(x.ravel(), y.ravel())
