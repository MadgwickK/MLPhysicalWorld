"""
Defines the gaussian process class along with different kernel fynctions.
"""
import numpy as np
from scipy.spatial.distance import cdist


def matern52_kernel(x_1, x_2, sigma_f, sigma_l):
    """
    Calculates the Matern 5/2 kernel for one or two given inputs
    Args:
        x_1 (ndarray): An ndarray of N dimensional data points.
        x_2 (ndarray or None): Same as x_1.
        sigma_f (float): Variance.
        sigma_l (float): Lenght scale.

    Returns: Covariance matrix.

    """
    dist = cdist(x_1, x_1 if x_2 is None else x_2)

    return (sigma_f ** 2) * (1 + np.sqrt(5) * dist / sigma_l +
                             5 * dist**2 /(3 * sigma_l**2)) * np.exp(-np.sqrt(5) * dist/sigma_l)


class GaussianProcess:
    """
    Gaussian Process class.

    This class implements a Gaussian process model with a user defined kernel and hyperparameters.

    Attributes:
        x_train (ndarray): Training data points (inputs).
        y_train (ndarray): Observed values at the training data points.
        kernel (function): Kernel function defining the covariance structure.
        sigma_l (float): Length scale hyperparameter for the kernel.
        sigma_f (float): Signal variance hyperparameter for the kernel.
        K (ndarray): Covariance matrix.
        L (ndarray): Cholesky decomposition of the training covariance matrix.
        alpha (ndarray): Precomputed weights for predictions.
    """
    def __init__(self, kernel, sigma_l, sigma_f, mean_function=None):
        self.x_train = None
        self.y_train = None

        self.kernel = kernel
        self.sigma_l = sigma_l
        self.sigma_f = sigma_f
        self.mean_function = mean_function or (lambda x: np.zeros(x.shape[0]))

        self.K = None
        self.L = None
        self.alpha = None

    def fit(self, x_train, y_train):
        """
        Fits the Gaussian Process.
        Args:
            x_train (ndarray): training data points.
            y_train (ndarray): f(x_train).
        """
        self.x_train = x_train
        self.y_train = y_train

        # Apply the mean function to the training data
        mean_train = self.mean_function(x_train)

        # Add small term to ensure numerical stability
        jitter = 1e-10
        # Covariance matrix for x_train
        self.K = (self.kernel(x_train, x_train, self.sigma_f, self.sigma_l)
             + jitter * np.eye(len(x_train)))

        # Cholesky decomposition of the covariance matrix of x_train
        self.L = np.linalg.cholesky(self.K)

        # Solve for alpha
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y_train - mean_train))

    def predict(self, x_test):
        """
        Calculates the mean and covariance matrix of given points.
        Args:
            x_test (ndarray): testing data points.

        Returns:
            mu_s (float): Predicted mean.
            cov_s (ndarray): Predicted covariance matrix.
        """
        # Covariance matrix between the training and testing data points.
        K_s = self.kernel(self.x_train, x_test, self.sigma_f, self.sigma_l)

        # Covariance matrix for the test data.
        K_ss = self.kernel(x_test, x_test, self.sigma_f, self.sigma_l)

        mean_test = self.mean_function(x_test)
        mu_s = mean_test + K_s.T.dot(self.alpha)

        var = np.linalg.solve(self.L, K_s)
        cov_s = K_ss - var.T.dot(var)
        return mu_s, cov_s
