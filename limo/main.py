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
        list(f.clip(self.nsamples) for f in self.features)
        self.rate = rate[-self.nsamples:]
        self.dt = dt

        # generate train/test indices
        self.train, self.test = holdout(batchify(self.nsamples, batch_size, True), frac_holdout)

        # keep track of stuff
        self.k = 0
        self.objective = list()
        self.test_obj = list()
        self.test_cc = list()

    def fit(self, num_epochs=5):

        for epoch in range(num_epochs):

            print('Epoch {:01d} of {:01d}'.format(epoch+1, num_epochs))

            # validate on test data
            self.score()

            # train
            list(map(self.feed, self.train))

        # score again
        self.score()

    def score(self):

        obj = []
        cc = []

        for batch in self.test:
            utot = self.predict(batch)
            rhat = np.exp(utot)
            obj.append(self.dt * np.nanmean(rhat - self.rate[batch] * utot))
            cc.append(pearsonr(rhat, self.rate[batch])[0])

        self.test_obj.append(np.nanmean(obj))
        self.test_cc.append(np.nanmean(cc))

    def predict(self, inds):

        # forward pass
        us = [f[inds] for f in self.features]

        # collect
        return np.sum(us, axis=0)

    def feed(self, inds):

        # compute the prediction
        utot = self.predict(inds)
        rhat = np.exp(utot)

        # backpropogate the error
        grads = [f(self.dt * (rhat - self.rate[inds])) for f in self.features]

        # save objective
        fobj = self.dt * np.nanmean(rhat - self.rate[inds] * utot)
        self.objective.append(fobj)

        # update iteration
        self.k += 1

        print('[{:d}] {}'.format(self.k, fobj))

    def __len__(self):
        return self.nsamples