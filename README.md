# Limo: generalized linear models
*Note: This package is just getting started. Check back later for more info.*

## About
The limo package is a set of tools for creating and fitting generalized linear
models (GLMs), specifically tuned for sensory neuroscience data and applications.

## Experiment
The `Experiment` object is useful as a container for holding sensory experiment
data. It exposes a number of helper utility functions.
```python
>>> ex = Experiment(stimulus, time, spikes, dt)
>>> ex.ste(nhist, cellidx) # generator for the spike triggered ensemble
>>> ex.stim_sliced(history) # sliced stimulus
```

## Feature
A `Feature` object exposes a couple of important methods, a `__call__` function
which takes a parameter array and convolves it with the feature, and a
`weighted_average` function which computes a weighted average of the feature
with the given input.

To create a feature, you must pass it a name (string) and a stimulus which has
dimensions `(..., ntimesteps)`

It must take as input a 
```python
>>> f = features.Stimulus('visual stimulus', upsampled_stimuli)
>>> f.shape # shape of the expected parameter array
>>> len(f) # number of samples
>>> proj = f(theta) # project / convolve the input with the feature
>>> sta = f.weighted_average(spikes) # compute a spike triggered average
```

## Objective
An `Objective` object is a container for the objective + gardient for a GLM with
the given features. It's `__call__` method returns a tuple containing the
objective and gradient.

```python
>>> f_df = Objective([feature1, feature2, ...], spikes, dt)
>>> f_df.theta_init # initial parameters
>>> objective, gardient = f_df(theta) # compute the objective and gradient at theta
```

You can plug `Objective` objects into the `descent` package:
```python
>>> descent.utils.check_grad(f_df, theta) # performs a numerical gradient check
>>> descent.optimize(descent.algorithms.rmsprop, f_df, f_df.theta_init)
```
