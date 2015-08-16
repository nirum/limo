import numpy as np

__all__ = ['Experiment']


class Experiment(object):
    """
    Sensory experiment
    """

    def __init__(self, stim, time, cells, dt):
        """
        TODO: Docstring for __init__.

        Parameters
        ----------
        stim : TODO
        time : TODO
        spikes : TODO

        Returns
        -------
        TODO

        """

        assert type(stim) == np.ndarray and stim.ndim == 3, \
            "Stimulus must be a 3 dimensionsal (space x space x time) array"

        assert type(time) == np.ndarray and time.ndim == 1, \
            "Time vector must be an one dimensional numpy array"

        self.stim = stim
        self.cells = cells
        self.time = time
        self.dt = dt

        spikes = list()
        for cell in self.cells:
            spikes.append(np.append(0, binspikes(cell, time=time)[0]))
        self.spikes = np.vstack(spikes)

    def __len__(self):
        return len(self.time)

    @property
    def tmax(self):
        return self.time[-1]

    @property
    def ncells(self):
        return self.spikes.shape[0]

    def stim_sliced(self, history):
        """Returns a view into the stimulus array"""
        return np.rollaxis(rolling_window(self.stim, history), 3, 2)

    def spike_history(self, history, offset=1):
        """Returns a view into the spikes array, offset by some amount"""
        arr = np.hstack((np.zeros((self.ncells,offset)), self.spikes[:, offset:]))
        return np.rollaxis(rolling_window(arr, history), 2, 1)

    def ste(self, stim_hist, ci):
        return (self.stim[..., (t-stim_hist):t].astype('float')
                for t in range(len(self))
                if self.spikes[ci,t] > 0 and t >= stim_hist)


def rolling_window(array, window):
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
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])

    """
    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
