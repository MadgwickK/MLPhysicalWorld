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
    pred_mag = FUNCTION(observed_times, [parameters[0], parameters[2]], parameters[1])
    loss = np.mean((magnifications - pred_mag) ** 2)
    return loss


def mae(observed_times, magnifications, parameters):
    pred_mag = FUNCTION(observed_times, [parameters[0], parameters[2]], parameters[1])
    loss = np.mean(np.abs(magnifications - pred_mag))
    return loss


def log_cosh(observed_times, magnifications, parameters):
    pred_mag = FUNCTION(observed_times, [parameters[0], parameters[2]], parameters[1])
    loss = np.mean(np.log(np.cosh(magnifications - pred_mag)))
    return loss


def log_likelihood(observed_times, magnifications, parameters, mag_err):
    pred_mag = FUNCTION(observed_times, [parameters[0], parameters[2]], parameters[1])
    term1 = (len(magnifications)/2) * np.sum(np.log(2*np.pi*(mag_err**2)))
    term2 = 0.5 * np.sum(((magnifications - pred_mag) / mag_err)**2)
    loss = term1 + term2
    loss_error = np.sqrt(2*term2)
    return loss, loss_error
