import numpy as np
from .features import Feature
from .utils import batchify, holdout
from scipy.stats import pearsonr


class PoissonGLM:

    def __init__(self, features, rate, dt, batch_size=1000, frac_holdout=0.1):

        # list of features
        self.features = features

        # clip lengths and the given rate
        self.nsamples = min(map(len, self.features))
        list(f.clip(self.nsamples) for f in features)
        self.rate = rate[-self.nsamples:]
        self.dt = dt

        # generate train/test indices
        self.train, self.test = holdout(batchify(self.nsamples, batch_size, True), frac_holdout)

        # keep track of stuff
        self.k = 0
        self.epoch = 0
        self.objective = list()
        self.test_obj = list()
        self.test_cc = list()

    def fit(self, num_epochs=5):

        for epoch in range(num_epochs):

            self.epoch += 1

            # validate on test data
            cc = []
            for batch in self.test:
                utot = self.predict(batch)
                rhat = np.exp(utot)
                self.test_obj.append(self.dt * np.nanmean(rhat - self.rate[batch] * utot))
                self.test_cc.append(pearsonr(rhat, self.rate[batch])[0])

            for batch in self.train:
                self.train(batch)

    def predict(self, inds):

        # forward pass
        us = [f[inds] for f in features]

        # collect
        return = np.sum(us, axis=0)

    def train(self, inds):

        # compute the prediction
        utot = self.predict(inds)
        rhat = np.exp(utot)

        # backpropogate the error
        grads = [f(self.dt * (rhat - self.rate[inds])) for f in features]

        # save objective
        self.objective.append(self.dt * np.nanmean(rhat - self.rate[inds] * utot))

        # update iteration
        self.k += 1
