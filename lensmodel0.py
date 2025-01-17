import numpy as np


def get_mag(u):
    """Computes magnification for given u value"""
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))


def get_r_E(M, d_L, d_S):
    """Computes Einstein Radius, r_E. Inputs: Mass of lens, M ; Distance to lens, d_L; Distance to source, d_S"""
    return np.sqrt(4 * (6.67e-11) * M * d_L * (d_S - d_L) / (9e16 * d_S))


def get_u(t, r_E, v_T, u_min=1, t_0=0):
    """Computes u at a certain time, assuming constant transverse velocity. Inputs: time, t; Einstein Radius, r_E;
    Transverse velocity v_T; minimum u value, u_min set at 0 by default; time of peak, t_0 set at 0 by default
    """
    return np.sqrt(u_min**2 + (v_T * (t - t_0) / r_E) ** 2)


def mean_function_theta(t, theta):
    """A mean function with no predicted degeneracy.
    Theta parameters:
    t_E: Einstein time in units of days
    u_min: Max magnitude of the microlensing event
    t_0: Central peak of the microlensing event"""

    t_E = theta[0] * 24 * (60**2)
    u_min = theta[1]
    t = t * 24 * (60**2)
    t_0 = theta[2] * 24 * (60**2)

    us = np.sqrt(u_min**2 + ((t - t_0) / t_E) ** 2)
    return get_mag(us)


def noisy_data_calc(low, upper, theta, noise, number, t_0=0):
    """Generates a noisy lightcurve specified using the given parameters. Returns randomly selected t values and
    their corresponding magnification.
    low: Lower time bound of the dataset, in units of days. Must be negative
    upper: Upper time bound of the dataset, in units of days. Must be positive
    theta: Parameters of the lightcurve model - t_E, u_min, t_0
    noise: Standard deviation of the gaussian distribution from which noise is sampled
    number: Number of samples to be selected.
    t_0: Peak of lightcurve. Default 0

    ---- returns ----
    ts: Array of t values generated randomly
    mags: Array of noisy magnification values
    """

    ts = np.random.uniform(low, upper, number)
    mags = mean_function_theta(ts, theta) + np.random.normal(0, noise, number)
    return ts, mags


def parallax_data_calc(low, upper, theta_s, noise_s, no_s, theta_e, noise_e, no_e):
    """Generates two lightcurves of the same microlensing event: measured from space, and from earth. There will be small
    differences between the two corresponding to the parallax.

    Parameters:
        low (float): Lower bound of time interval
        upper (float): Upper bound of time interval
        theta_s (ndarray): Parameters of space lightcurve - t_E, u_0, t_0
        noise_s (float): Standard deviation of Gaussian noise of space magnification measurements
        no_s (int): Number of datapoints in space lightcurve
        theta_e (ndarray): Parameters of Earth lightcurve - t_E, u_0, t_0
        noise_e (float): Standard deviation of Gaussian noise of Earth magnification measurements
        no_e (int): Number of datapoints in Earth lightcurve

    Returns:
        t_s (ndarray): array of time values for space measurements, in units of days
        mag_s (ndarray): array of magnification values for space measurements
        t_e (ndarray): array of time values for Earth measurements, in units of days
        mag_e (ndarray): array of magnification values for Earth measurements
    """
    t_s = np.linspace(low, upper, no_s) + np.random.normal(0, 1, no_s)
    mag_s = mean_function_theta(t_s, theta_s) + np.random.normal(0, noise_s, no_s)
    t_e = np.linspace(low, upper, no_e) + np.random.normal(0, 0.1, no_e)
    mag_e = mean_function_theta(t_e, theta_e) + np.random.normal(0, noise_e, no_e)
    return t_s, mag_s, t_e, mag_e


def parallax_resolver(theta_s, theta_e, error_s, error_e, distance=1):
    """To be ran after finding the parameters from the space-based lightcurve and earth-based lightcurve.

    Parameters:
        theta_s (ndarray): Parameters of the lightcurve in space - t_E, u_0, t_0
        theta_e (ndarray): Parameters of the lightcurve on Earth - t_E, u_0, t_0
        error_s (ndarray): Errors on the space lightcurve parameters
        error_e (ndarray): Errors on the earth lightcurve parameters
        distance (float): Distance between Earth and the space telescope, units of AU. Default = 1

    Returns:
        r_E (ndarray): Einstein Radius, in units of AU, and associated error
        v_T (ndarray): Transverse velocity of object, in units of km s^-1, and associated error
    """

    delta_t_0 = (theta_s[2] - theta_e[2]) ** 2
    delta_t_0_err = 2 * np.sqrt(delta_t_0 * (error_s[2] ** 2 + error_e[2] ** 2))
    delta_u_0 = (theta_s[1] - theta_e[1]) ** 2
    delta_u_0_err = 2 * np.sqrt(delta_u_0 * (error_s[1] ** 2 + error_e[1] ** 2))

    r_E = distance / (np.sqrt(delta_t_0 + delta_u_0))
    r_E_err = r_E * np.abs(0.5 * np.sqrt(delta_t_0_err**2 + delta_u_0_err**2))
    t_E = (theta_s[0] + theta_e[0]) / 2
    t_E_err = 1 / 2 * np.sqrt(error_s[0] ** 2 + error_e[0] ** 2)

    v_T = r_E * 1.496e8 / (t_E * 60**2 * 24)
    v_T_err = v_T * np.sqrt((r_E / r_E_err) ** 2 + (t_E / t_E_err) ** 2)
    if v_T > 3e5:
        print(
            "This combination gives a transverse velocity greater than the speed of light."
        )
    return np.array([r_E, r_E_err]), np.array([v_T, v_T_err])
