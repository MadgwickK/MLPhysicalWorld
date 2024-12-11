"""
Different sampling methods for the Bayesian optimisation implementation in bayes_opt.py.
Within all the functions, we swap d_L and d_S if d_S<=d_L.
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

    # Ensures d_L <= d_S by swapping them
    for i in range(num_samples):
        if samples[i, 1] <= samples[i, 0]:
            samples[i, 0], samples[i, 1] = samples[i, 1], samples[i, 0]

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
    bounds_array = np.array(list(self.bounds.values()))
    dim = bounds_array.shape[0]
    self.x_samples = np.empty((0, dim))
    for _ in range(num_samples):

        # Array for our sample values
        sample = []

        # Array to store d_L and d_S
        param_values = {
            'd_L': self.bounds['d_L'][0],
            'd_S': self.bounds['d_S'][1]
        }

        for param, (lower, upper) in self.bounds.items():

            if param == 'd_L':

                # Generates two numbers within the bounds of d_S and d_L
                num_1 = np.random.uniform(lower, upper)
                num_2 = np.random.uniform(self.bounds['d_S'][0], self.bounds['d_S'][1])

                # Assigns the larger number to d_S and the smaller number to d_L
                param_values['d_L'] = np.minimum(num_1, num_2)
                param_values['d_S'] = np.maximum(num_1, num_2)
                sample.append(param_values['d_L'])

            elif param == 'd_S':

                # Appends calculated value for d_S
                sample.append(param_values['d_S'])

            else:
                sample.append(np.random.uniform(lower, upper))

        # Stores sample to x_samples
        sample = np.array(sample)
        samples = np.vstack((self.samples, sample))
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
    swap_indices = samples[:, 1] <= samples[:, 0]
    samples[swap_indices, 0], samples[swap_indices, 1] = (samples[swap_indices, 1],
                                                          samples[swap_indices, 0])

    return samples
