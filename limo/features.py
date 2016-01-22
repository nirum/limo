import numpy as np
from descent import algorithms
from .utils import inner

__all__ = ['Feature']


class Feature:

    def __init__(self, stimulus, learning_rate=1e-4, l2=1e-3, algorithm='adam'):
        """
        Initializes a 'feature'

        Parameters
        ----------
        stimulus : array_like
            The first dimension indexes the sample, while the rest indicate
            the feature space of the stimulus

        learning_rate : float, optional
            The learning rate of the optimizer (Default: 1e-4)

        l2 : float, optional
            Strength of the l2 regularization (Default: 1e-3)

        algorithm : string, optional
            Which algorithm to use, taken from descent.algorithms (Default: 'adam')

        """

        ndim = len(stimulus.shape)
        assert ndim <= 18, "Too many dimensions!"

        # l2 regularization
        self.l2 = l2

        # use the Hessian trick
        self.hessian = False

        # store the stimulus
        self.stimulus = stimulus

        # initial parameters
        theta_init = 1e-4 * np.random.randn(*self.stimulus.shape[1:])
        self.theta = theta_init.copy()

        # pick out the algorithm to use
        self.optimizer = getattr(algorithms, algorithm)(theta_init, lr=learning_rate)

        # create the einsum notation strings for projecting stimuli onto the
        # feature and for averaging a rate over the stimuli
        letters = 'tijklmnopqrstuvwxyz'
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
            alpha = 1. / mu  # (1./self.l2 + 1./mu)
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
    def ndim(self):
        return self.stimulus.ndim

    @property
    def shape(self):
        return self.theta.shape

    def clip(self, length):
        """Clips this feature"""
        self.stimulus = self.stimulus[-length:, ...]

    def __len__(self):
        return self.stimulus.shape[0]
