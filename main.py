from turtledemo.chaos import coosys

import numpy as np
from gaussian_process import GaussianProcess, matern52_kernel
from lensmodel import noisy_data_calc
from bayes_opt import BayesianOptimisation, expected_improvement
from samplers import gaussian_sampling
from objectives import mse
import matplotlib.pyplot as plt
from lensmodel import mean_function_theta

X, Y = noisy_data_calc(-40, 40, [20, 0.5], 0.05, 50, t_0=0)

parameter_bounds = {
    't_E':      [0.01, 700],    # days
    't_0':      [-5, 5],        # days (placeholder, updated in the code)
    'u_min':    [0, 4]          # unitless
}


def fit_mag(observed_times, magnifications, mean_function=None):
    """
    Fits a Gaussian process to the observed times and magnifications.
    Args:
        observed_times (ndarray): Times observed.
        magnifications (ndarray): Magnifications observed.
        mean_function (callable): Function encoding prior information about the funciton being fit.

    Returns:
        mag_gaussian (class): Fitted Gaussian.
    """

    # Fits the Gaussian process to the observed data using the mean function
    mag_gaussian = GaussianProcess(kernel=matern52_kernel, sigma_l=2, sigma_f=1,
                                   mean_function=mean_function)
    mag_gaussian.fit(observed_times.reshape(-1, 1), magnifications, noise_variance=0.01)
    return mag_gaussian


def create_mean_function(gaussian_process, observed_times, magnifications):
    """
    Creates a mean function for the objective function using the Gaussian process fitted to the
    observed data by calculating the mean square error between the fitted Gaussian and the real
    data.
    Args:
        gaussian_process (class): Gaussian fitted to the observed data and magnifications
        observed_times (ndarray): Times observed.
        magnifications (ndarray): Magnifications observed.

    Returns:
        constant_mean_function (function): Mean function for the objective function.
    """
    # Predict the magnifications at the observed times
    pred_mag, cov = gaussian_process.predict(observed_times.reshape(-1, 1), noise_variance=0.01)

    # Calculate the mean squared error between the predicted magnifications and the observed values
    loss = np.mean((magnifications - pred_mag) ** 2)

    # Define the mean function for the objective's GP
    def constant_mean_function(x):
        return loss

    return constant_mean_function


def estimate_t_0(gaussian_process, bounds):
    """
    Estimates t_0 using the Gaussian process fitted on the real data.
    Args:
        gaussian_process (class): Gaussian fitted to the observed data and magnifications.
        bounds (dict): Dictionary of bounds for the parameters being fitted.

    Returns:
        t_0 (float): Estimated t_0.
        mag_peak (float): Estimated magnification peak.
    """
    # Create a fine grid of time values over the time space
    t_space = np.linspace(bounds['t_0'][0], bounds['t_0'][1], 10000)

    # Predict magnifications at each point of the fine grid
    pred_mag, cov = gaussian_process.predict(t_space.reshape(-1, 1), noise_variance=0.01)
    errors = np.sqrt(np.diag(cov))

    # Find the maximum predicted magnification
    max_mag_index = np.argmax(pred_mag)
    mag_peak = pred_mag[max_mag_index]

    # Calculate the error on the predicted magnification at t_0
    error_mag_peak = errors[max_mag_index]

    # Find t_0 at which magnification is maximised
    t_0 = t_space[max_mag_index]

    return t_0, mag_peak, error_mag_peak


def estimate_u_min(mag_peak, mag_peak_error):
    """
    Computes u_min exactly given a predicted magnification.
    Args:
        mag_peak (float): Estimated magnification peak.
        mag_peak_error (float): Estimated magnification peak error.

    Returns:
        u_min (float): Estimated u_min.
        u_min_error (float): Estimated error on u_min.
    """
    # Computes a predicted u_min
    u_min_sqrt = 2 * ((np.abs(mag_peak) / np.sqrt(mag_peak ** 2 - 1)) - 1)
    u_min = np.sqrt(u_min_sqrt)

    # Compute the error on u_min
    u_min_der = -1 / (np.power(mag_peak ** 2 - 1, 3/2) * u_min)
    u_min_error = np.abs(u_min_der * mag_peak_error)

    return u_min, u_min_error


def estimate_t_E(observed_times, magnifications, t_0, u_min, u_min_error):
    """
    Estimates t_E with errors from the Gaussian process fitted on the real data. This uses the fact
    that magnification curves have a Gaussian bell-like shape.
    Args:
        observed_times (ndarray): Times observed.
        magnifications (ndarray): Magnifications observed.
        t_0 (float): Estimated t_0.
        u_min (float): Estimated u_min.
        u_min_error (float): Estimated error on u_min.

    Returns:
        t_E (float): Estimated t_E.
        t_E_error (float): Estimated error on t_E.
    """
    # Calculate bound on magnifications such that the calculated t_E exists
    u_min_fact = ((u_min**2 / 2) + 1)**2
    max_mag = np.sqrt(u_min_fact / (u_min_fact - 1)) - 1e-4 # jitter value for numerical stability
    valid_mask = (magnifications > 1) & (magnifications < max_mag)

    # Use bound on magnifications to eliminate all values in pred_mag and t_space where t_E doesn't exist
    observed_times = observed_times[valid_mask]
    magnifications = magnifications[valid_mask]

    # Predict t_E for each predicted magnification
    denominator = 4 - (u_min ** 2) * (u_min ** 2 + 4) * (magnifications ** 2 - 1)
    numerator = 2 * magnifications * np.sqrt(magnifications ** 2 - 1) + (u_min ** 2 + 2) * (magnifications ** 2 - 1)
    t_E_samples_sq = ((observed_times - t_0)**2) * (numerator / denominator)

    # Makes sure there is no negative root
    valid_t_E_mask = t_E_samples_sq > 0
    t_E_samples_sq = t_E_samples_sq[valid_t_E_mask]
    observed_times = observed_times[valid_t_E_mask]
    magnifications = magnifications[valid_t_E_mask]
    denominator = denominator[valid_t_E_mask]
    numerator = numerator[valid_t_E_mask]

    # Calculate t_E for every observation
    t_E_samples = np.sqrt(t_E_samples_sq)

    # Propagate error on u_min for every t_E
    t_E_samples_der = ((observed_times - t_0)**2)*(((u_min ** 2) * (4*magnifications*np.sqrt((magnifications ** 2) - 1) + (magnifications ** 2 - 1) * u_min ** 2) + 4*numerator + 4) / ((denominator ** 2) * t_E_samples))
    t_E_sample_errors = t_E_samples_der * u_min_error

    # Predict t_E as the average of all predicted t_E values
    t_E = np.mean(t_E_samples)
    t_E_error = np.sqrt(np.std(t_E_samples)**2 + (np.sqrt(np.sum(t_E_sample_errors**2)) / len(t_E_samples))**2)

    return t_E, t_E_error


def bootstrapping(observed_times, magnifications, bounds, mean_function=None):
    """
    Randomly selects data to predict the parameters 20 times such that an error is predicted using
    bootstrapping.
    Args:
        observed_times (ndarray): Times observed.
        magnifications (ndarray): Magnifications observed.
        bounds (dict): Dictionary of bounds for the parameters being fitted.
        mean_function (callable): Function encoding prior information about the funciton being fit.

    Returns:
        t_E (float): Estimated t_E.
        t_E_error (float): Estimated error on t_E.
        t_0 (float): Estimated t_0.
        t_0_error (float): Estimated error on t_0.
        u_min (float): Estimated u_min.
        u_min_error (float): Estimated u_min_error.
    """
    t_0_samples, t_E_samples, u_min_samples = [], [], []
    u_min_error_samples, t_E_error_samples = [], []
    for i in range(20):
        # Randomly selects datapoints
        indices = np.random.choice(len(observed_times), len(observed_times), replace=True)
        sampled_times = observed_times[indices]
        sampled_magnifications = magnifications[indices]

        # Fit Gaussian process to datapoints
        mag_gaussian = fit_mag(sampled_times, sampled_magnifications, mean_function)

        # Predict parameters for the chosen data using Gaussian process
        t_0, mag_peak, mag_peak_error = estimate_t_0(mag_gaussian, bounds)
        t_0_samples.append(t_0)

        u_min, u_min_error = estimate_u_min(mag_peak, mag_peak_error)
        u_min_samples.append(u_min)
        u_min_error_samples.append(u_min_error)

        t_E, t_E_error = estimate_t_E(observed_times, magnifications, t_0, u_min, u_min_error)
        t_E_samples.append(t_E)
        t_E_error_samples.append(t_E_error)

    # Take mean and standard deviations from sampled parameters
    t_E = np.mean(t_E_samples)
    t_E_error_samples = np.array(t_E_error_samples)
    t_E_error = np.sqrt(np.std(t_E_samples)**2 + (np.sqrt(np.sum(t_E_error_samples**2)) / len(t_E_samples))**2)

    t_0 = np.mean(t_0_samples)
    t_0_error = np.std(t_0_samples)

    u_min = np.mean(u_min_samples)
    u_min_error_samples = np.array(u_min_error_samples)
    u_min_error = np.sqrt(np.std(u_min_samples)**2 + (np.sqrt(np.sum(u_min_error_samples**2)) / len(u_min_samples))**2)
    return t_E, t_E_error, t_0, t_0_error, u_min, u_min_error



def main(observed_times, magnifications, bounds):
    mean_function = None
    sigma = 1

    # Change bounds on t_0 before estimating
    bounds['t_0'] = [np.min(observed_times), np.max(observed_times)]

    # Estimate parameters using bootstrapping
    t_E, t_E_error, t_0, t_0_error, u_min, u_min_error = bootstrapping(observed_times,
                                                                       magnifications, bounds,
                                                                       mean_function)

    print('t_0 = ', t_0, '+/-', sigma*t_0_error)
    print('u_min = ', u_min, '+/- ', sigma*u_min_error)
    print('t_E = ', t_E, '+/- ', sigma*t_E_error)

    # Changes bounds to take into account predictions
    bounds['t_0'] = [t_0 - sigma*t_0_error, t_0 + sigma*t_0_error]
    bounds['t_E'] = [t_E - sigma*t_E_error, t_E + sigma*t_E_error]
    bounds['u_min'] = [u_min - sigma*u_min_error, u_min + sigma*u_min_error]

    parameter_samples = []
    for i in range(10):
        # Define surrogate Gaussian process
        gp = GaussianProcess(kernel=matern52_kernel, sigma_l=1, sigma_f=1)

        # Define Bayesian optimisation
        optimiser = BayesianOptimisation(surrogate=gp, acquisition=expected_improvement,
                                         objective=mse, bounds=parameter_bounds,
                                         sampler=gaussian_sampling)

        # Fit for parameters using the defined Bayesian optimiser
        optimiser.fit(observed_times, magnifications, 300)

        # Plot regret and results
        optimiser.regret_plot()
        # optimiser.plot_best_param()

        # Append found parameters to the list of all predictions
        parameter_samples.append(optimiser.x_samples[optimiser.current_best_index])

    best_parameters = np.mean(parameter_samples, axis=0)
    best_parameter_errors = np.std(parameter_samples, axis=0)
    print('t_0: ', best_parameters[1], '+/- ', best_parameter_errors[1])
    print('u_min: ', best_parameters[2], '+/- ', best_parameter_errors[2])
    print('t_E: ', best_parameters[0], '+/- ', best_parameter_errors[0])
    times = np.linspace(np.min(observed_times), np.max(observed_times), 10000)
    mags = mean_function_theta(times, [best_parameters[0], best_parameters[2]], best_parameters[1])
    plt.plot(times, mags, color='blue')
    plt.scatter(observed_times, magnifications, color='red')
    plt.show()


main(X, Y, parameter_bounds)
