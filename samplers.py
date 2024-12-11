"""
Different sampling methods for the Bayesian optimisation implementation in bayes_opt.py.
Within all the functions, we swap d_L and d_S if d_S<=d_L.
"""
import numpy as np
from scipy.stats import qmc

def swap(samples, dim1, dim2):
    """
    Swaps any sample with dim2 <= dim1
    Args:
        samples (ndarray): Samples.
        dim1 (int): Dimension we want to make larger.
        dim2 (int): Dimension we want to make smaller.

    Returns:
        samples (ndarray): Swapped samples.
    """
    swap_indices = samples[:, dim2] <= samples[:, dim1]
    samples[swap_indices, dim1], samples[swap_indices, dim2] = (samples[swap_indices, dim2],
                                                          samples[swap_indices, dim1])
    return samples


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

    # Ensures d_L <= d_S by swapping them
    samples = swap(samples, 0, 1)

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

        # Ensures d_L <= d_S
        d_L = np.minimum(sample[0], sample[1])
        d_S = np.maximum(sample[0], sample[1])
        sample[0], sample[1] = d_L, d_S

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
    sampler = qmc.Sobol(d=dim)
    samples = sampler.random(n=num_samples)

    # Scale the samples using the previously defined bounds
    samples = qmc.scale(samples, bounds_array[:, 0], bounds_array[:, 1])

    # Ensures d_L <= d_S by swapping them
    # Ensures d_L <= d_S by swapping them
    samples = swap(samples, 0, 1)

    return samples
