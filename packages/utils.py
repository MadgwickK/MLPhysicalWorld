import numpy as np
from gaussian_process import GaussianProcess, matern52_kernel
from bayes_opt import BayesianOptimisation, expected_improvement
from samplers import gaussian_sampling
from objectives import log_likelihood
import matplotlib.pyplot as plt
from parameter_estimation import estimate_params
from lensmodel import mean_function_theta


def read_and_convert(file_name, I_0, f_s):
    """
    This function reads a file for microlens event (.dat) and converts the magnitude to magnification.
    -------------------------
    Parameters:
    file_name (str): The name of the file to read.
    I_0 (float): The reference magnitude. (can be found in the data webpage and zip file)
    -------------------------
    Returns:
    data (DataFrame): The data read from the file with an additional column for magnification.
    """
    data = np.loadtxt(file_name)
    times = data[:, 0]
    magnitudes = data[:, 1]
    mag_errors = data[:, 2]
    delta_I = magnitudes - I_0
    magnifications = ((10 ** (delta_I / -2.5) - 1) / f_s) + 1
    magnification_errors = magnifications * np.log(10) * (mag_errors / 2.5)
    # magnifications += 1 - np.min(magnifications)

    # mag_peak = np.max(magnifications)
    # threshold = 1 + (mag_peak - 1) * 0.0
    # mask = magnifications > threshold
    # magnifications = magnifications[mask]
    # times = times[mask]
    # magnification_errors = magnification_errors[mask]

    return times, magnifications, magnification_errors


def plot_final_params(observed_times, magnifications, best_parameters, mags_error=None):
    """
    Plot the predicted magnification function and the observed data.
    Args:
        observed_times (ndarray): Times observed.
        magnifications (ndarray): Magnifications observed.
        best_parameters (list): Parameters being plotted.
        mags_error (ndarray): Errors on the observed magnifications.

    """
    # Create fine mesh of times from the first observation to the last observation
    times = np.linspace(np.min(observed_times), np.max(observed_times), 10000)

    # Calculate predicted magnifications
    theta = [best_parameters[0], best_parameters[2]]
    mags = mean_function_theta(times, theta, best_parameters[1])

    # Plot prediction with the observed data
    plt.plot(times, mags, color="blue")
    if mags_error is not None:
        plt.errorbar(observed_times, magnifications, mags_error, fmt=".", color="red")
    else:
        plt.scatter(observed_times, magnifications, color="red")
    plt.xlabel("Time / Heliocentric Julian Days")
    plt.ylabel("Magnification")
    plt.savefig(
        f"BO_plot/final_params{best_parameters[0]}_{best_parameters[1]}_{best_parameters[2]}.png"
    )
    plt.show()
