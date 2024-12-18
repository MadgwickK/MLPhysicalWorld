import numpy as np
from lensmodel import mean_function_theta

FUNCTION = mean_function_theta


def mse(observed_times, magnifications, parameters):
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
    pred_mag = FUNCTION(observed_times, parameters)
    loss = np.mean((magnifications - pred_mag) ** 2)
    return loss


def mae(observed_times, magnifications, parameters):
    pred_mag = FUNCTION(observed_times, parameters)
    loss = np.mean(np.abs(magnifications - pred_mag))
    return loss


def log_cosh(observed_times, magnifications, parameters):
    pred_mag = FUNCTION(observed_times, parameters)
    loss = np.mean(np.log(np.cosh(magnifications - pred_mag)))
    return loss
