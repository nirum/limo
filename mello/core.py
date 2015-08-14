"""
generalized linear models
"""

# from descent.utils import destruct, restruct
import numpy as np
from collections import defaultdict
from toolz import valmap, itemmap
from pyret.spiketools import binspikes

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

    def stim_sliced(self, history):
        """Returns a view into the stimulus array"""
        return rolling_window(self.stim, history)

    def spikes_sliced(self, history):
        """Returns a view into the spikes array"""
        return rolling_window(self.spikes, history)

    def ste(self, stim_hist, ci):
        return (self.stim[..., (t-stim_hist):t].astype('float')
                for t in range(len(self))
                if self.spikes[ci,t] > 0 and t >= stim_hist)

    def features(self, stim_hist=50, spk_hist=80):
        """Note: Slower than using np.einsum"""
        mint = max(stim_hist, spk_hist+1)
        for t in range(mint, len(self)):
            st = self.stim[..., (t-stim_hist):t]
            spk = self.spikes[..., (t-spk_hist-1):(t-1)]
            yield {'stim': st, 'spk': spk}
        raise StopIteration

class Objective(object):
    """
    Experiment agnostic
    """

    def __init__(self):
        pass
        # self.features = get_features
        # self.N = float(num_features)
        # self.dt = dt

        # # compute average feature
        # for idx, f in enumerate(self.features()):

            # if idx == 0:
                # tmp = f

            # else:
                # for k, v in f.items():
                    # tmp[k] += v

        # self.average_feature = valmap(lambda v: np.divide(v, N), tmp)

    def project(self, x, theta):
        return np.sum([np.tensordot(theta[k], x[k]) for k in theta.keys()])

    def __call__(self, theta):
        """TODO: Docstring for __call__.

        Parameters
        ----------
        theta : TODO

        Returns
        -------
        TODO

        """

        # np.einsum('ijtk,ijk->t', stim, theta) # 3
        # np.einsum('itk,ik->t', stim, theta) # 2

        avg_proj = self.project(self.average_feature, theta)

        total_rate = 0
        weighted_features = defaultdict(np.ndarray)

        for idx, f in enumerate(self.features()):
            r = np.exp(self.project(f, theta))
            total_rate += r

            if idx == 0:
                weighted_features = valmap(lambda v: np.multiply(r, v), f)

            else:
                for k, v in f.items():
                    weighted_features[k] += np.multiply(r, v)

        obj = (self.dt / self.N) * total_rate - avg_proj
        grad = itemmap(lambda k, v: (self.dt / self.N) * v - avg_proj[k], weighted_features)

        return obj, grad


def make_objective(ex, ci, stim_hist=40, spk_hist=80):

    mint = np.max(stim_hist, spk_hist+1)
    T = float(len(ex) - mint)

    stim_sliced = ex.stim_sliced(stim_hist)
    spk_sliced = ex.spikes_sliced(spk_hist+1)

    nspk = ex.spikes[ci]

    stim_avg = np.einsum('ijtk,t->ijk', stim_sliced, nspk[stim_hist:]) / float(len(ex) - stim_hist)
    spk_avg = np.einsum('itj,t->ij', spk_sliced, nspk[spk_hist:])[:,:-1] / float(len(ex) - spk_hist + 1)

    def f_df(theta):

        ustim = np.einsum('ijtk,ijk->t', stim_sliced, theta['stim'])
        uspk = np.einsum('itj,ij->t', spk_sliced, np.hstack((theta['spk'], np.zeros((theta['spk'].shape[0],1)))))
        rhat = np.exp(ustim + uspk)

        obj = (ex.dt / float(rhat.size)) * np.sum(rhat) - np.dot(theta['stim'].ravel(), stim_avg.ravel()) - np.dot(theta['spk'].ravel(), spk_avg.ravel())
        grad_stim = (ex.dt / T) * np.einsum('ijtk,t->ijk', stim_sliced, rhat)

    return f_df


# USAGE:
# ex = Experiment(stim, time, spikes, dt)
# f_df = Objective(ex, stim_hist=50, spk_hist=80)
# xhat = descent.optimize(descent.rmsprop, f_df, x0, callbacks=cb, maxiter=100)


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
