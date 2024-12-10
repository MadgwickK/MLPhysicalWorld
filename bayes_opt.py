"""
Implementation of Bayesian optimisation.
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm, qmc
from lensmodel import mean_function_theta


parameter_bounds = {
    'd_L':                      [0, 10000],
    'd_S':                      [0, 10000],
    'v_M_ratio':                [0, 1000],
    'u_min':                    [0, 4]
}

XI = 0.1 # Exploration parameter for expected improvement.
FUNCTION = mean_function_theta
X = np.random.uniform(-30, 30, 20)
Y = mean_function_theta(X, [70, 400, 100, 1])
DATA = np.column_stack((X, Y))


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

    # Predict mean and variance at x, making sure to avoid division by zero.
    mean, cov = surrogate.predict(candidates)
    sigma = np.sqrt(np.diag(cov))
    sigma = np.maximum(sigma, 1e-9)

    # Compute diff and Z
    diff = y_best - mean - XI
    Z = diff / sigma

    # Compute expected improvement, making sure it is non-negative.
    exp_imp = diff * norm.cdf(Z) + sigma * norm.pdf(Z)
    exp_imp = np.maximum(exp_imp, 0.0)
    return exp_imp


class BayesianOptimisation:
    """
    Bayesian Optimisation class.

    This class implements bayesian optimisation with user deined surrogate, acquisition and
    objective functions, as well as user defined hyperparameters.

    Attributes:
        surrogate (class): Surrogate model.
        acquisition (function): Acquisition function.
        objective (function): Objective function.
        bounds (dict): Parameter bounds.
        iteration_n (int): Number of iterations.
        x_obs (ndarray): Observed data.
        y_obs (ndarray): Observed data.
        x_samples (ndarray): Previously sampled points.
        y_samples (ndarray): Previously sampled points.
        current_best (float): Current best point.
    """
    def __init__(self, surrogate, acquisition, objective, bounds, iteration_n):
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.objective = objective
        self.bounds = bounds
        self.iteration_n = iteration_n

        self.x_obs = None
        self.y_obs = None
        self.x_samples = []
        self.y_samples = []
        self.current_best = None

    def initial_sampling(self, num_samples=10):
        """
        Samples a number of initial parameters to start the bayesian optimisation
        with some parameter space exploration.
        Args:
            num_samples (int): Number of intial samples.
        """
        for _ in range(num_samples):
            # Randomly generates values for all parameters, ensuring d_S >= d_L
            d_L = np.random.uniform(self.bounds['d_L'][0], self.bounds['d_L'][1])
            d_S = np.random.uniform(d_L, self.bounds['d_S'][1])
            v_M_ratio = np.random.uniform(self.bounds['v_M_ratio'][0], self.bounds['v_M_ratio'][1])
            u_min = np.random.uniform(self.bounds['u_min'][0], self.bounds['u_min'][1])

            # Adds the sample to the list x_samples
            sample = [d_L, d_S, v_M_ratio, u_min]
            self.x_samples = np.vstack((self.x_samples, sample))

            # Calculates the objective at the sample and adds it to the list y_sample
            y_sample = self.objective(self.x_obs, self.y_obs, sample)
            self.y_samples = np.append(self.y_samples, y_sample)

        # Turns the sample lists into ndarrays
        self.x_samples = np.array(self.x_samples)
        self.y_samples = np.array(self.y_samples)

    def latin_hypercube_sampling(self, num_samples=100):
        """
        Implementation of latin hypercube sampling.
        Args:
            num_samples (int): Number of samples.

        Returns:
            samples (ndarray): Samples.
        """
        # Turn directory into array of bounds.
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

    def propose_location(self):
        """
        Proposes a new sample using latin hypercube sampling and the acquisition function.
        Returns:
            best_candidate (ndarray): Best candidate point.
        """
        # Generate candidates using latin hypercube sampling
        candidates = self.latin_hypercube_sampling()

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
        self.current_best = np.min(self.y_samples)

    def fit(self, x_obs, y_obs):
        """
        Runs the bayesian optimisation algorithm for given observations.
        Args:
            x_obs (ndarray): Observed data.
            y_obs (ndarray): Observed data.
        """
        self.x_obs = x_obs
        self.y_obs = y_obs

        # Samples initial points to kickstart optimisation
        self.initial_sampling()

        # Runs algorithm for n iterations
        for _ in range(self.iteration_n):
            self.update()
