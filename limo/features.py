import numpy as np
from descent.algorithms import adam

__all__ = ['Feature']


class Feature:

    def __init__(self, stimulus, lr=1e-4, l2=1e-3):

        ndim = len(stimulus.shape)
        assert ndim <= 6, "Too many dimensions!"

        self.l2 = l2

        self.stimulus = stimulus
        self.ndim = self.stimulus.ndim

        theta_init = 1e-4 * np.random.randn(*self.stimulus.shape[1:])
        self.optimizer = adam(theta_init, learning_rate=lr)
        self.theta = self.optimizer.send(None)

        letters = 'tijklmn'
        self.einsum_proj = letters[:self.ndim] + ',' + \
            letters[1:self.ndim] + '->' + letters[0]

        self.einsum_avg = letters[:self.ndim] + ',' + \
            letters[0] + '->' + letters[1:self.ndim]

    def __getitem__(self, inds):
        """
        Forward projection using the given indices

        """

        self.minibatch = self.stimulus[inds]
        return np.einsum(self.einsum_proj, self.minibatch, self.theta)

    def __call__(self, err, active=True):
        """
        Backpropogate an error signal

        """

        # compute the gradient
        gradient = np.einsum(self.einsum_avg, self.minibatch, err) / float(err.size)

        # add l2 regularization
        gradient += self.l2 * self.theta

        # clear memory
        self.minibatch = None

        # gradient update
        if active:
            self.theta = self.optimizer.send(gradient)

        return gradient

    @property
    def shape(self):
        return self.theta.shape

    def clip(self, length):
        """Clips this feature"""
        self.stimulus = self.stimulus[-length:, ...]

    def __len__(self):
        return self.stimulus.shape[0]
