"""
Implementation of Bayesian optimisation to find parameters given observed microlensing data.
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from itertools import accumulate
from samplers import sobol_sampling

from lensmodel import mean_function_theta
from gaussian_process import GaussianProcess, matern52_kernel


parameter_bounds = {
    't_E':      [0.01, 700],    # days
    'u_min':    [0, 4],         # unitless
    't_0':      [-5, 5]         # days
}

# Exploration parameter for expected improvement
XI = 0.5

# Function being fit by the parameters
FUNCTION = mean_function_theta
X = np.random.uniform(-30, 30, 20)
Y = mean_function_theta(X, [70, 400, 100, 1])


def objective_function(observed_times, magnifications, parameters):
    """
    Calculates the mean squared error of some function given the real data and a set of the
    function´s parameters.
    Args:
        observed_times (ndarray): Times observed.
        magnifications (ndarray): Magnifications observed.
        parameters (ndarray): Model parameters.

    Returns:
        loss (float): Mean squared error.
    """
    y_pred = FUNCTION(observed_times, parameters)
    loss = mean_squared_error(magnifications, y_pred)
    return loss


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
        sampler (function): Sampler function.
        bounds (dict): Parameter bounds.
        observed_times (ndarray): Times observed.
        magnifications (ndarray): Magnifications observed.
        iteration_n (int): Number of iterations.
        x_samples (ndarray): Previously sampled parameters.
        y_samples (ndarray): Previously sampled losses.
        current_best_index (int): Index of the current best point in the x and y sample arrays.
    """
    def __init__(self, surrogate, acquisition, objective, sampler, bounds):
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.objective = objective
        self.bounds = bounds
        self.sampler = sampler

        self.observed_times = None
        self.magnifications = None
        self.iteration_n = None

        bounds_array = np.array(list(self.bounds.values()))
        dim = bounds_array.shape[0]
        self.x_samples = np.empty((0, dim))
        self.y_samples = np.empty((0, 1))
        self.current_best_index = None

    def propose_location(self):
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

    def update(self):
        """
        Proposes a sampling location, computes the objective, and adds the sample to the lists of
        all samples. Also updates the current best sample.
        """
        # Propose next sampling location
        x_next = self.propose_location()

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
        if self.bounds['t_0'] is not None:
            self.bounds['t_0'] = [np.minimum(self.observed_times), np.maximum(self.observed_times)]

        # Sample initial points to kickstart optimisation
        self.x_samples = np.vstack((self.x_samples, self.sampler(self, 2 ** 4)))
        for sample in self.x_samples:
            y_sample = self.objective(self.observed_times, self.magnifications, sample)
            self.y_samples = np.append(self.y_samples, y_sample)

        # Start surrogate model
        self.surrogate.fit(self.x_samples, self.y_samples)

        # Run algorithm for n iterations
        for _ in range(self.iteration_n):
            self.update()

    def plot_best_param(self):
        # Call best found parameters
        best_parameters = self.x_samples[self.current_best_index]

        # Calculate the magnification function for these best parameters
        t = np.linspace(-40, 40, 10000)
        magnification = FUNCTION(t, best_parameters)

        plt.plot(t, magnification, color='blue')
        plt.scatter(self.observed_times, self.magnifications, color='red')
        plt.show()

    def regret_plot(self):
        current_best_losses = list(accumulate(-self.y_samples, max))
        plt.plot(current_best_losses)
        plt.show()


start = time.time()

gp = GaussianProcess(kernel=matern52_kernel, sigma_l=2, sigma_f=1)

optimiser = BayesianOptimisation(surrogate=gp, acquisition=expected_improvement,
                                 objective=objective_function, sampler=sobol_sampling,
                                 bounds=parameter_bounds)
optimiser.fit(X, Y, 1000)
end = time.time()

optimiser.regret_plot()
optimiser.plot_best_param()
print(end - start)
print(optimiser.x_samples[optimiser.current_best_index])
