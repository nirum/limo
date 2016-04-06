import numpy as np
from .utils import batchify, holdout
from scipy.stats import pearsonr
from os.path import expanduser, join

__all__ = ['PoissonGLM']


class PoissonGLM:

    def __init__(self, features, rate, dt, batch_size=1000, frac_holdout=0.0):
        """
        Initialize a Poisson GLM

        Parameters
        ----------
        features : list
            list of `Feature` objects (see features.py)

        rate : array_like
            An array of responses / firing rate

        dt : float
            The time step (in seconds)

        batch_size : int
            The batch size to use when loading stimuli. Only this many samples
            of the stimulus are loaded at a time.

        frac_holdout : float
            Fraction of the stimuli to hold out as a test set.

        """

        # keep track of the features and the time step
        self.features = features
        self.dt = dt

        # clip all of the features and the rate to be the same length
        self.nsamples = min(map(len, self.features))
        [f.clip(self.nsamples) for f in self.features]
        self.rate = rate[-self.nsamples:]

        # generate train/test indices
        batches = batchify(self.nsamples, int(batch_size), True)
        self.train, self.test = holdout(batches, frac_holdout)

        # keep track of the current iteration and other metrics
        self.k = 0
        self.objective = list()
        self.test_obj = list()
        self.test_cc = list()

    def fit(self, num_epochs=5):
        """Learn the optimal GLM parameters"""

        for epoch in range(num_epochs):

            print('Epoch {:01d} of {:01d}'.format(epoch + 1, num_epochs))

            # validate on test data
            self.score()

            # train
            list(map(self.feed, self.train))

        # score again
        self.score()

    def score(self):
        """Tests the model using log-likelihood and correlation coefficient"""

        obj = []
        cc = []

        for batch in self.test:
            utot = self.predict(batch)
            rhat = np.exp(utot)
            obj.append(self.dt * np.nanmean(rhat - self.rate[batch] * utot))
            cc.append(pearsonr(rhat, self.rate[batch])[0])

        self.test_obj.append(obj)
        self.test_cc.append(cc)

    def predict(self, inds):
        """Predicted firing rate""""

        # forward pass
        us = [f[inds] for f in self.features]

        # collect
        return np.sum(us, axis=0)

    def feed(self, inds):
        """Forward pass"""

        # compute the prediction
        utot = self.predict(inds)
        rhat = np.exp(utot)
        mu = rhat.mean()

        # backpropogate the error
        [f(self.dt * (rhat - self.rate[inds]), mu) for f in self.features]

        # save objective
        fobj = self.dt * np.nanmean(rhat - self.rate[inds] * utot)
        self.objective.append(fobj)

        # update iteration
        self.k += 1

        print('[{:d}] {}'.format(self.k, fobj))

    def save(self, fname, basedir='~/Dropbox/data/GLMs'):
        """Save parameters to a .npz file"""
        theta = [f.theta for f in self.features]
        np.savez(expanduser(join(basedir, fname)),
                 test_obj=self.test_obj,
                 test_cc=self.test_cc,
                 obj=self.objective, params=theta)

    def __len__(self):
        return self.nsamples
