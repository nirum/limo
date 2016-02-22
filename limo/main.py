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

        # we need to clip all of the features and the rate to be the same length
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

    def fit(self, num_epochs, monitor=None):
        """Fits the parameters of the GLM"""

        iteration = 0

        for epoch in range(num_epochs):

            print('Epoch {:01d} of {:01d}'.format(epoch + 1, num_epochs))

            for inds in self.train:

                # train on batch
                fobj, r_train, rhat_train = self.feed(inds)

                # performs validation, updates performance plots, saves to dropbox
                if (monitor is not None) and (iteration % monitor.save_every == 0):

                    # test
                    rhat_test = self.predict(*monitor.testdata)
                    monitor.save(epoch, iteration, r_train, rhat_train, rhat_test)

                    # save parameters
                    self.save('epoch{}_iteration{}.npz', basedir=monitor.datadir)

                # update
                print('{}\tLoss: {}'.format(iteration, fobj))
                iteration += 1

    def score(self):

        obj = []
        cc = []

        for batch in self.test:
            utot = self._project(batch)
            rhat = np.exp(utot)
            obj.append(self.dt * np.nanmean(rhat - self.rate[batch] * utot))
            cc.append(pearsonr(rhat, self.rate[batch])[0])

        self.test_obj.append(obj)
        self.test_cc.append(cc)

    def predict(self, *Xs):
        """rate prediction given the stimulus and rates"""
        utot = np.sum([f._project(x) for x, f in zip(Xs, self.features)], axis=0)
        rhat = np.exp(utot)
        return rhat

    def _project(self, inds):

        # forward pass
        us = [f[inds] for f in self.features]

        # collect
        return np.sum(us, axis=0)

    def feed(self, inds):

        # compute the prediction
        utot = self._project(inds)
        rhat = np.exp(utot)
        mu = rhat.mean()

        # backpropogate the error
        [f(self.dt * (rhat - self.rate[inds]), mu) for f in self.features]

        # save objective
        fobj = self.dt * np.nanmean(rhat - self.rate[inds] * utot)
        self.objective.append(fobj)

        # update iteration
        self.k += 1

        return fobj, self.rate[inds], rhat

    def save(self, fname, basedir='~/Dropbox/data/GLMs'):
        theta = [f.theta for f in self.features]
        np.savez(expanduser(join(basedir, fname)), test_obj=self.test_obj, test_cc=self.test_cc,
                 obj=self.objective, params=theta)

    def __len__(self):
        return self.nsamples
