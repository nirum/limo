import limo
import numpy as np

def generate_problem(T):

    X = (np.random.rand(T,100) * 255).astype('uint8')
    r = np.random.rand(T)

    return X, r


if __name__ == '__main__':

    T = 1e6

    X, r = generate_problem(T)

    f1 = limo.Convolutional('stim', X, history=40)

    obj = limo.Objective([f1], r, 1e-2)
    # f1.weighted_average(r[:-40])

    foo, bar = obj(obj.theta_init)
