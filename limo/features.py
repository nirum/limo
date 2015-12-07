import numpy as np
from descent.algorithms import adam

__all__ = ['Feature']


def inner(x,y):
    return np.inner(x.ravel(), y.ravel())


class Feature:

    def __init__(self, stimulus, lr=1e-4, l2=1e-3):

        ndim = len(stimulus.shape)
        assert ndim <= 6, "Too many dimensions!"

        self.l2 = l2
        self.hessian = False

        self.stimulus = stimulus
        self.ndim = self.stimulus.ndim

        theta_init = 1e-4 * np.random.randn(*self.stimulus.shape[1:])
        self.optimizer = adam(theta_init, lr=lr)
        self.theta = theta_init.copy()

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

    def __call__(self, err, mu, active=True):
        """
        Backpropogate an error signal

        Parameters
        ----------
        err : array_like

        mu : float
            Mean firing rate

        """

        # compute the gradient
        gradient = np.einsum(self.einsum_avg, self.minibatch, err) / float(err.size)

        # add l2 regularization
        gradient += self.l2 * self.theta

        if self.hessian:
            # works for white noise stimuli
            wtw = inner(self.theta, self.theta)
            wtg = inner(self.theta, gradient)
            alpha = 1. / mu #(1./self.l2 + 1./mu)
            beta = wtg / ((1. + wtw) * mu)
            print('a={}, b={}'.format(alpha, beta))
            gradient = alpha * gradient - self.theta * beta

        # clear memory
        self.minibatch = None

        # gradient update
        if active:
            self.theta = self.optimizer(gradient)

        return gradient

    @property
    def shape(self):
        return self.theta.shape

    def clip(self, length):
        """Clips this feature"""
        self.stimulus = self.stimulus[-length:, ...]

    def __len__(self):
        return self.stimulus.shape[0]
