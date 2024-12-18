import numpy as np
from gaussian_process import GaussianProcess, rbf_kernel
from lensmodel import noisy_data_calc
from bayes_opt import BayesianOptimisation, expected_improvement
from samplers import sobol_sampling
from objectives import mse


X, Y = noisy_data_calc(-70, 70, [40, 1], 0.0, 100, t_0=0)

parameter_bounds = {
    't_E':      [0.01, 700],    # days
    't_0':      [-5, 5],        # days (placeholder, updated in the code)
    'u_min':    [0, 4]          # unitless
}


def fit_mag(observed_times, magnifications):
    """
    Fits a Gaussian process to the observed times and magnifications.
    Args:
        observed_times (ndarray): Times observed.
        magnifications (ndarray): Magnifications observed.

    Returns:
        mag_gaussian (class): Fitted Gaussian.
    """

    # Fits the Gaussian process to the observed data using the mean function
    mag_gaussian = GaussianProcess(kernel=rbf_kernel, sigma_l=20, sigma_f=1)
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
    Estimates t_0 with errors from the Gaussian process fitted on the real data.
    Args:
        gaussian_process (class): Gaussian fitted to the observed data and magnifications.
        bounds (dict): Dictionary of bounds for the parameters being fitted.

    Returns:
        t_0 (float): Estimated t_0.
        mag_peak (float): Estimated magnification peak.
        error_t_0 (float): Estimated error on t_0.
        error_mag_peak (float): Estimated error on the magnification peak.
    """
    # Create a fine grid of time values over the time space
    t_space = np.linspace(bounds['t_0'][0], bounds['t_0'][1], 10000)

    # Predict magnifications at each point of the fine grid
    pred_mag, cov = gaussian_process.predict(t_space.reshape(-1, 1), noise_variance=0.01)
    errors = np.sqrt(np.diag(cov))

    # Find the maximum predicted magnification
    max_mag_index = np.argmax(pred_mag)
    mag_peak = pred_mag[max_mag_index]

    # Find t_0 at which magnification is maximised
    t_0 = t_space[max_mag_index]

    # Finds the second derivative of the Gaussian process using central differences method
    second_deriv_pred_mag = np.gradient(np.gradient(pred_mag, t_space), t_space)

    # Calculate the error on the predicted magnification at t_0
    error_mag_peak = errors[max_mag_index]

    # Propagates uncertainty using the fact that delta(x) = sqrt(delta(y) /  |f''(x)|) for y = f(x)
    error_t_0 = np.sqrt(error_mag_peak / np.abs(second_deriv_pred_mag[max_mag_index]))

    return t_0, mag_peak, error_t_0, error_mag_peak


def estimate_t_E(gaussian_process, bounds, mag_peak):
    """
    Estimates t_E with errors from the Gaussian process fitted on the real data.
    Args:
        gaussian_process (class): Gaussian fitted to the observed data and magnifications.
        bounds (dict): Dictionary of bounds for the parameters being fitted.
        mag_peak (float): Estimated magnification peak.

    Returns:
        t_E (float): Estimated t_E.
        t_E_error (float): Estimated error on t_E.
    """
    # Create a fine grid of time values over the time space
    t_space = np.linspace(bounds['t_0'][0], bounds['t_0'][1], 10000)

    # Predict magnifications at each point of the fine grid
    pred_mag, cov = gaussian_process.predict(t_space.reshape(-1, 1), noise_variance=0.01)
    errors = np.sqrt(np.diag(cov))

    # Predicted magnification half maximum
    half_max_mag = (mag_peak - np.min(pred_mag)) / np.sqrt(2) + np.min(pred_mag)

    # Calculate the indices where pred_max = half_max_mag,
    # taking into account that they won't be exactly equal
    tolerance = 1e-4
    diff_mag_half_max = np.abs(pred_mag - half_max_mag)
    indices = np.where(diff_mag_half_max <= np.min(diff_mag_half_max + tolerance))[0]

    # Estimate t_E as being the full width half maximum
    t_fwhm_0 = t_space[indices[-1]]
    t_fwhm_1 = t_space[indices[0]]

    t_E = np.abs(t_fwhm_0 - t_fwhm_1)

    # Finds the derivative of the Gaussian process using central differences method
    derivative_mag = np.gradient(pred_mag, t_space)

    # Finds errors on the times the magnification is at the full width half maximum
    err_mag_fwhm_0 = errors[indices[-1]]
    err_mag_fwhm_1 = errors[indices[0]]

    err_t_fwhm_0 = err_mag_fwhm_0 / np.abs(derivative_mag[indices[-1]])
    err_t_fwhm_1 = err_mag_fwhm_1 / np.abs(derivative_mag[indices[0]])

    # Finds error on the predicted t_E
    error_t_E = np.sqrt(err_t_fwhm_0 ** 2 + err_t_fwhm_1 ** 2)

    return t_E, error_t_E


def estimate_u_min(mag_peak, error_mag_peak):
    """
    Computes u_min exactly given a predicted magnification
    Args:
        mag_peak (float): Estimated magnification peak.
        error_mag_peak (float): Estimated error on the magnification peak.

    Returns:
        u_min (float): Estimated u_min.
        u_min_error (float): Estimated error on u_min.
    """
    # Computes magnification peak squared
    mag_squared = mag_peak ** 2

    # Computes a predicted u_min
    u_min_sqrt = 2 * (-mag_squared + np.sqrt(mag_squared * (mag_squared - 1)) + 1) / (mag_squared - 1)
    u_min = np.sqrt(u_min_sqrt)

    # Computes the first derivative of u_min
    u_min_derivative = -(mag_peak ** 3) / (np.power((mag_squared * (mag_squared - 1)), 3/2) * u_min)

    # Computes the error on the predicted u_min
    u_min_error = np.abs(u_min_derivative) * error_mag_peak

    return u_min, u_min_error


def main(observed_times, magnifications, bounds):
    # Fit Gaussian process to observed data
    mag_gaussian = fit_mag(observed_times, magnifications)

    # Updates the bounds on t_0
    bounds['t_0'] = [np.min(observed_times), np.max(observed_times)]

    # Predicts the maximum magnification and t_0 with errors
    t_0, mag_peak, error_t_0, error_mag_peak  = estimate_t_0(mag_gaussian, bounds)

    # Predicts t_E with errors
    t_E, error_t_E = estimate_t_E(mag_gaussian, bounds, mag_peak)

    # Predicts u_min with errors
    u_min, u_min_error = estimate_u_min(mag_peak, error_mag_peak)

    # Changes bounds to take into account predictions
    bounds['t_0'] = [t_0 - error_t_0, t_0 + error_t_0]
    bounds['t_E'] = [t_E - error_t_E, t_E + error_t_E]
    bounds['u_min'] = [u_min - u_min_error, u_min + u_min_error]
    print(bounds)

    # Define surrogate Gaussian process
    gp = GaussianProcess(kernel=rbf_kernel, sigma_l=2, sigma_f=1)

    # Define Bayesian optimisation
    optimiser = BayesianOptimisation(surrogate=gp, acquisition=expected_improvement, objective=mse,
                                     bounds=parameter_bounds, sampler=sobol_sampling)

    # Fit for parameters using the defined Bayesian optimiser
    optimiser.fit(X, Y, 1000)

    # Plot regret and results
    optimiser.regret_plot()
    optimiser.plot_best_param()

    # Print best parameters and the associated loss
    print('best parameters:', optimiser.x_samples[optimiser.current_best_index])
    print('best error:', optimiser.y_samples[optimiser.current_best_index])


main(X, Y, parameter_bounds)
