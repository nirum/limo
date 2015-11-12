"""
Shared utilities

"""
import numpy as np

__all__ = ['batchify', 'epochify']


def epochify(num_epochs, t, n, randomize):

    for rep in range(num_epochs):

        print('Epoch {:01d} of {:01d}'.format(rep+1, num_epochs))

        yield from batchify(t, n, randomize)


def batchify(t, n, randomize=True):

    inds = np.arange(t)
    if randomize:
        np.random.shuffle(inds)

    while len(inds) > 0:

        yield inds[:n]
        inds = np.delete(inds, slice(n))
