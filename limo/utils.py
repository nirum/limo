"""
Shared utilities

"""
import numpy as np

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


def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to
    window : int
        Size of rolling window

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    -------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3),1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])

    """
    if time_axis == 0:
        array = array.T
    elif time_axis == -1:
        pass
    else:
        raise ValueError('Time axis must be first or last')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr
