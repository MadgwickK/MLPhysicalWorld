"""
Implementation of Bayesian optimisation.
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from itertools import accumulate
import samplers as samp

from lensmodel import mean_function_theta
from gaussian_process import GaussianProcess, matern52_kernel


parameter_bounds = {
    'd_L':                      [0, 10000], # In parsecs
    'd_S':                      [0, 10000], # In parsecs
    'v_M_ratio':                [0, 1000],  # km s^{-1} (solar mass)^{-1}
    'u_min':                    [0, 4]      # unitless?
}

XI = 0.1                        # Exploration parameter for expected improvement
FUNCTION = mean_function_theta  # Function being fit by the parameters
X = np.random.uniform(-30, 30, 20)
Y = mean_function_theta(X, [70, 400, 100, 1])

def objective_function(x_obs, y_obs, parameters):
    """
    Calculates the mean squared error of some function given the real data and a set of the
    function´s parameters.
    Args:
        x_obs (ndarray): Observed data.
        y_obs (ndarray): Observed data.
        parameters (ndarray): Model parameters.

    Returns:
        loss (float): Mean squared error.
    """
    y_pred = FUNCTION(x_obs, parameters)
    loss = mean_squared_error(y_obs, y_pred)
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
        x_obs (ndarray): Observed data.
        y_obs (ndarray): Observed data.
        iteration_n (int): Number of iterations.
        x_samples (ndarray): Previously sampled points.
        y_samples (ndarray): Previously sampled points.
        current_best_index (int): Index of the current best point in the x and y sample arrays.
    """
    def __init__(self, surrogate, acquisition, objective, sampler, bounds):
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.objective = objective
        self.bounds = bounds
        self.sampler = sampler

        self.x_obs = None
        self.y_obs = None
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
        # Generate candidates using latin hypercube sampling
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
        y_next = self.objective(self.x_obs, self.y_obs, x_next)

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
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.iteration_n = iteration_n

        # Sample initial points to kickstart optimisation
        self.x_samples = np.vstack((self.x_samples, self.sampler(self, 10)))
        for sample in self.x_samples:
            y_sample = self.objective(self.x_obs, self.y_obs, sample)
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
        plt.scatter(self.x_obs, self.y_obs, color='red')
        plt.show()

    def regret_plot(self):
        current_best_losses = list(accumulate(-self.y_samples, max))
        plt.plot(current_best_losses)
        plt.show()


start = time.time()

gp = GaussianProcess(kernel=matern52_kernel, sigma_l=2, sigma_f=1)

optimiser = BayesianOptimisation(surrogate=gp, acquisition=expected_improvement,
                                 objective=objective_function, sampler=samp.sobol_sampling,
                                 bounds=parameter_bounds)
optimiser.fit(X, Y, 100)
end = time.time()

optimiser.regret_plot()
optimiser.plot_best_param()
print(end - start)
