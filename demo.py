"""
Fitting a GLM using limo
"""
import numpy as np
import matplotlib.pyplot as plt
import limo


def simulate():
    """Simulates data from a toy model"""

    ndim = 20
    nsamples = 10000

    # 'true' parameters
    theta = np.sin(np.linspace(0, 2 * np.pi, ndim))
    theta /= np.linalg.norm(theta)

    # generate responses
    X = np.random.randn(nsamples, ndim)
    r = np.exp(X.dot(theta))

    return X, r, theta


def main():
    """Fits a model to simulated data"""

    # generate data from the toy model
    dt = 0.1
    X, r, theta_star = simulate()

    # build a GLM
    feature = limo.Feature(X, learning_rate=1e-3, l2=0.0)
    glm = limo.PoissonGLM([feature], r, dt, batch_size=1000, frac_holdout=0.2)

    # fit the parameters
    glm.fit(num_epochs=100)

    # plot the objective of the training and holdout sets
    fig = plt.figure(1)
    ax = fig.add_subplot(121)
    ax.plot(glm.objective)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('negative log-likelihood')
    ax.set_title('Train')

    ax = fig.add_subplot(122)
    ax.plot(np.squeeze(glm.test_obj).mean(axis=1))
    ax.set_xlabel('Epoch')
    ax.set_title('Hold-out (validation)')

    # plot the parameters
    theta_hat = glm.features[0].theta
    theta_hat /= np.linalg.norm(theta_hat)
    plt.figure(2)
    plt.plot(theta_star, '-', color='gray', label='True parameters')
    plt.plot(theta_hat, '--', color='lightcoral', label='Estimated parameters')
    plt.legend(loc=0, fancybox=True, frameon=True)

    plt.show()
    plt.draw()

    return glm


if __name__ == '__main__':
    glm = main()
