"""
Shared utilities

"""
import numpy as np
from pyret.filtertools import rolling_window

__all__ = ['batchify', 'epochify', 'holdout', 'preprocess']


def epochify(num_epochs, t, n, randomize, holdout=0.2):

    for rep in range(num_epochs):

        print('Epoch {:01d} of {:01d}'.format(rep+1, num_epochs))

        yield from batchify(t, n, randomize)


def holdout(batches, frac=0.1):

    batches = list(batches)
    num_holdout = int(np.round(len(batches) * frac))

    test = batches[:num_holdout]
    train = batches[num_holdout:]

    return train, test


def batchify(t, n, randomize=True):

    inds = np.arange(t)
    if randomize:
        np.random.shuffle(inds)

    while len(inds) > 0:

        yield inds[:n]
        inds = np.delete(inds, slice(n))


def preprocess(stimulus, history, zscore=(0.,1.)):
    """
    Preprocess stimulus array

    """

    stim = np.array(stimulus).astype('float')
    stim -= zscore[0]
    stim /= zscore[1]
    return rolling_window(stim, history, time_axis=0)
