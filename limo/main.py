import numpy as np
from .features import Feature
from .utils import epochify, batchify

# f = Feature(...)

# u = f[inds]
# rhat = np.exp(u)
# obj = np.mean(dt * rhat - robs[inds] * u)
# grad = f.update(dt * rhat - robs[inds])

for k, batch in enumerate(epochify(num_epochs, len(f), batch_size, randomize)):

    u = f[batch]
    rhat = np.exp(u)
    grad = f(rhat * 1e-2 - robs[batch])

# class PoissonGLM:

    # def __init__(self, features):

            # for f in features:


# class Foo:

    # def __init__(self, X, theta_init):

        # self.X = X
        # self.optimizer = adam(theta, learning_rate=1e-2)
        # self.theta = self.optimizer.send(None)
        # self.ndim = self.theta.ndim

    # def __getitem__(self, inds):

        # # load and save this minibatch for later
        # self.minibatch = self.X[inds]

        # # return projection (u)
        # return np.tensordot(self.minibatch, self.theta, self.ndim)

    # def update(self, err):

        # # compute the gradient
        # gradient = np.tensordot(err, self.minibatch, 1) / float(err.size)

        # # update weights
        # self.theta = self.optimizer.send(gradient)

        # # gradient
        # return gradient

    # def __len__(self):
        # return self.X.shape[0]


# def feature():

    # def __iter__(self):
        # while True:
            # print('>> Waiting for indices')
            # inds = yield
            # print('>> Got indices: ', inds)
            # print('>> Loading data')
            # x = self.data[inds]
            # u = np.tensordot(x, self.theta, self.theta.ndim)
            # err = yield u
            # print('>> Got error: ', err)
            # self.theta = self.optimizer.send(np.tensordot(err, x, 1))
            # print('>> Updated theta')

