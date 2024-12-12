"""
Different sampling methods for the Bayesian optimisation implementation in bayes_opt.py.
"""
import numpy as np
from scipy.stats import qmc


def latin_hypercube_sampling(self, num_samples=2 ** 10):
    """
    Implementation of latin hypercube sampling.
    Args:
        self: Bayesian optimisation class attributes.
        num_samples (int): Number of samples.

    Returns:
        samples (ndarray): Samples.
    """
    # Turn directory into array of bounds
    bounds_array = np.array(list(self.bounds.values()))

    # Sample points from 0 to 1 in 4 dimensions using Latin Hypercube Sampling
    dim = bounds_array.shape[0]
    sampler = qmc.LatinHypercube(d=dim)
    samples = sampler.random(n=num_samples)

    # Scale the samples using the previously defined bounds
    samples = qmc.scale(samples, bounds_array[:, 0], bounds_array[:, 1])

    return samples


def uniform_random(self, num_samples=2 ** 10):
    """
    Implementation of uniform random sampling.
    Args:
        self: Bayesian optimisation class attributes.
        num_samples (int): Number of samples.

    Returns:
        samples (ndarray): Samples.
    """
    # Turns Bounds directory into a numpy array
    bounds_array = np.array(list(self.bounds.values()))

    # Creates an empty numpy array for all the samples
    dim = bounds_array.shape[0]
    samples = np.zeros((num_samples, dim))

    for i in range(num_samples):
        # Samples a random point using user defined bounds
        sample = np.random.uniform(bounds_array[:, 0], bounds_array[:, 1])

        # Adds sample to the numpy array of all samples
        samples[i] = sample

    return samples


def sobol_sampling(self, num_samples=2 ** 10):
    """
    Implementation of Sobol sampling.
    Args:
        self: Bayesian optimisation class attributes.
        num_samples (int): Number of samples.

    Returns:
        samples (ndarray): Samples.
    """

    # Turn directory into array of bounds
    bounds_array = np.array(list(self.bounds.values()))

    # Sample points from 0 to 1 in 4 dimensions using Sobol sampling
    dim = bounds_array.shape[0]
    sampler = qmc.Sobol(d=dim, scramble=True)
    samples = sampler.random(n=num_samples)

    # Scale the samples using the previously defined bounds
    samples = qmc.scale(samples, bounds_array[:, 0], bounds_array[:, 1])

    return samples
