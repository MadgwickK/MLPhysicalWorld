import numpy as np

def get_mag(u):
    '''Computes magnification for given u value'''
    return (u**2 + 2)/(u*np.sqrt(u**2 + 4))

def get_r_E(M,d_L,d_S):
    '''Computes Einstein Radius, r_E. Inputs: Mass of lens, M ; Distance to lens, d_L; Distance to source, d_S'''
    return np.sqrt(4*(6.67e-11)*M*d_L*(d_S-d_L)/(9e16 * d_S))

def get_u(t,r_E, v_T,u_min=1,t_0=0):
    '''Computes u at a certain time, assuming constant transverse velocity. Inputs: time, t; Einstein Radius, r_E;
    Transverse velocity v_T; minimum u value, u_min set at 0 by default; time of peak, t_0 set at 0 by default'''
    return np.sqrt(u_min**2 + (v_T*(t-t_0)/r_E)**2)

def mean_function(t,theta):
    '''A mean function with no predicted degeneracy.
    Theta parameters:
    t_E: Einstein time in units of days
    u_min: Max magnitude of the microlensing event
    t_0: Central peak of the microlensing event'''

    t_E = theta[0]*24*(60**2)
    u_min = theta[1]
    t = t*24*(60**2)
    t_0 = theta[2]*24*(60**2)

    us = np.sqrt(u_min**2 + ((t-t_0)/t_E)**2)
    return get_mag(us)

def noisy_data_calc(low,upper,theta,noise,number,t_0=0):
    '''Generates a noisy lightcurve specified using the given parameters. Returns randomly selected t values and
    their corresponding magnification.
    low: Lower time bound of the dataset, in units of days. Must be negative
    upper: Upper time bound of the dataset, in units of days. Must be positive
    theta: Parameters of the lightcurve model - t_E, u_min, t_0
    noise: Standard deviation of the gaussian distribution from which noise is sampled
    number: Number of samples to be selected.
    t_0: Peak of lightcurve. Default 0'''

    ts = np.random.uniform(low,upper,number)
    mags = mean_function_theta(ts,theta) + np.random.normal(0,noise,number)
    return ts, mags
