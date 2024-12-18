"""
Implementation of Bayesian optimisation to find parameters given observed microlensing data.
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from itertools import accumulate
from samplers import uniform_random, sobol_sampling
from lensmodel import mean_function_theta
from gaussian_process import GaussianProcess, rbf_kernel
from objectives import mse


parameter_bounds = {
    't_E':      [0.01, 100],    # days 700
    't_0':      [-5, 5],        # days
    'u_min':    [0, 4]          # unitless
}

# Exploration parameter for expected improvement
XI = 0.2


FUNCTION = mean_function_theta


def create_mean_function(observed_times, magnifications):
    mag_gaussian = GaussianProcess(kernel=rbf_kernel, sigma_l=2, sigma_f=1)
    mag_gaussian.fit(observed_times.reshape(-1, 1), magnifications)
    pred_mag, cov = mag_gaussian.predict(observed_times.reshape(-1, 1))
    loss = np.mean((magnifications - pred_mag) ** 2)

    def constant_mean_function(x):
        return loss

    return constant_mean_function


def expected_improvement(candidates, y_samples, surrogate):
    """
    Calculates the expected improvement of x given previous observations and a surrogate model.
    Specifically, this is expected improvement for a minimisation task.
    Args:
        candidates (ndarray): Candidate points.
        y_samples (ndarray): Previously sampled points,
        surrogate (class): Surrogate model.

    Returns:
        exp_imp (float): Expected improvement.
    """
    # Current best observation
    y_best = np.min(y_samples)

    # Predict mean and variance at x, making sure to avoid division by zero
    mean, cov = surrogate.predict(candidates)
    sigma = np.sqrt(np.diag(cov))
    sigma = np.maximum(sigma, 1e-9)

    # Compute diff and Z
    diff = y_best - mean - XI
    Z = diff / sigma

    # Compute expected improvement, making sure it is non-negative
    exp_imp = diff * norm.cdf(Z) + sigma * norm.pdf(Z)
    exp_imp = np.maximum(exp_imp, 0.0)
    return exp_imp


class BayesianOptimisation:
    """
    Bayesian Optimisation class.

    This class implements bayesian optimisation with user deined surrogate, acquisition, sampling,
    and objective functions, as well as user defined hyperparameters.

    Attributes:
        surrogate (class): Surrogate model.
        acquisition (function): Acquisition function.
        objective (function): Objective function.
        bounds (dict): Parameter bounds.
        sampler (function): Sampler function, if none is specified then grid sampling is used.
        observed_times (ndarray): Times observed.
        magnifications (ndarray): Magnifications observed.
        iteration_n (int): Number of iterations.
        x_samples (ndarray): Previously sampled parameters.
        y_samples (ndarray): Previously sampled losses.
        current_best_index (int): Index of the current best point in the x and y sample arrays.
    """
    def __init__(self, surrogate, acquisition, objective, bounds, sampler=None):
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.objective = objective
        self.bounds = bounds
        self.sampler = sampler

        self.observed_times = None
        self.magnifications = None
        self.iteration_n = None
        self._grid_cache = None

        bounds_array = np.array(list(self.bounds.values()))
        dim = bounds_array.shape[0]
        self.x_samples = np.empty((0, dim))
        self.y_samples = np.empty((0, 1))
        self.current_best_index = None


    def _propose_location(self):
        """
        Proposes a new sample with the user defined sampler function and the acquisition function.
        Returns:
            best_candidate (ndarray): Best candidate point.
        """
        # Generate candidates using the given sampling function
        candidates = self.sampler(self, num_samples=(2 ** 10))

        # Calculate the expected improvement of each sample
        exp_imp = self.acquisition(candidates, self.y_samples, self.surrogate)

        # Find index for best candidate
        best_index = np.argmax(exp_imp)

        # Find best candidate
        best_candidate = candidates[best_index]

        return best_candidate

    def _update(self):
        """
        Proposes a sampling location, computes the objective, and adds the sample to the lists of
        all samples. Also updates the current best sample.
        """
        # Propose next sampling location
        x_next = self._propose_location()

        # Evaluate the objective at new location
        y_next = self.objective(self.observed_times, self.magnifications, x_next)

        # Update observed samples
        self.x_samples = np.vstack((self.x_samples, x_next))
        self.y_samples = np.append(self.y_samples, y_next)

        # Update surrogate model
        self.surrogate.fit(self.x_samples, self.y_samples)

        # Update current best
        self.current_best_index = np.argmin(self.y_samples)

    def fit(self, x_obs, y_obs, iteration_n):
        """
        Runs the bayesian optimisation algorithm for given observations.
        Args:
            x_obs (ndarray): Observed data.
            y_obs (ndarray): Observed data.
            iteration_n (int): Number of iterations.
        """
        self.observed_times = x_obs
        self.magnifications = y_obs
        self.iteration_n = iteration_n

        # Update bounds on t_0 if t_0 is a bound
        if 't_0' in self.bounds:
            self.bounds['t_0'] = [np.min(self.observed_times), np.max(self.observed_times)]

        # Sample initial points to kickstart optimisation using uniform random sampling
        self.x_samples = np.vstack((self.x_samples, uniform_random(self, 16)))
        for sample in self.x_samples:
            y_sample = self.objective(self.observed_times, self.magnifications, sample)
            self.y_samples = np.append(self.y_samples, y_sample)

        # Calculate current best
        self.current_best_index = np.argmin(self.y_samples)

        # Start surrogate model
        self.surrogate.fit(self.x_samples, self.y_samples)

        # Run algorithm for n iterations
        for _ in range(self.iteration_n):
            self._update()

    def plot_best_param(self):
        # Call best found parameters
        best_parameters = self.x_samples[self.current_best_index]

        # Calculate the magnification function for these best parameters
        t = np.linspace(-70, 70, 10000)
        magnification = FUNCTION(t, best_parameters)

        plt.plot(t, magnification, color='blue')
        plt.scatter(self.observed_times, self.magnifications, color='red')
        plt.show()

    def regret_plot(self):
        current_best_losses = list(accumulate(-self.y_samples, max))
        plt.plot(current_best_losses)
        plt.show()
