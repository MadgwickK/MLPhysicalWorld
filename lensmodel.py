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

def mean_function(t,M,d_L,d_S,v_T,u_min=1,t_0=0):
    '''Computes the mean magnification function from set parameters. 
    t is an array of times in units of days, 
    M the mass of the lens in units of solar masses, 
    d_L and d_S the distance to the lens and source respectively in units of parsecs, 
    v_T the transverse velocity of the lens with respect to the source in units of km s^-1.'''
    if d_L>d_S:
        raise ValueError("Distance to lens, d_L, cannot exceed distance to source, d_S")
    
    #Adjusting to new units
    tnew = t*24*(60**2)
    t_0new = t_0*24*(60**2)
    Mnew = M*1.99e30
    d_Lnew = d_L*3.086e16
    d_Snew = d_S*3.086e16
    v_Tnew = v_T*1e3

    r_E = get_r_E(Mnew,d_Lnew,d_Snew)
    us = get_u(tnew,r_E,v_Tnew,u_min,t_0new)
    return get_mag(us)

def mean_function_with_planet(t,M,d_L,d_S,v_T,M_p,r_P,u_min=1,t_0=0):
    '''Computes the mean magnification function with a planet present. All parameters same as original mean function.
    t is an array of times in units of days, 
    M the mass of the lens in units of solar masses, 
    d_L and d_S the distance to the lens and source respectively in units of parsecs, 
    v_T the transverse velocity
    of the lens with respect to the source in units of km s^-1.
    Additional planetary parameters: Mass of planet, M_p is in Jupiter masses. 
    Transverse distance between star and
    planet, r_P, is in units of Astronomical Units and can be positive or negative.
    We assume the orbital radius is negligible compared to the source/lens distance.'''

    M_pnew = M_p*1.898e27/1.99e30
    t_0p = r_P*1.496e8/(v_T*24*(60**2))
    
    star_mag = mean_function(t,M,d_L,d_S,v_T,u_min,t_0)
    planet_mag = mean_function(t,M_pnew,d_L,d_S,v_T,u_min,t_0=t_0p)
    return star_mag+planet_mag-1

def mean_function_theta(t,theta,t_0=0):
    '''t is time in units of days
    Theta is a tuple of all parameters, in the following order:
    d_L: Distance to source, units of parsecs
    d_S: Distance to lens, units of parsecs
    v_M_ratio: Ratio between transverse velocity of lens and square root of the lens mass. Units of km s^-1 M_sun^-1/2
    The two variables are degenerate so can't be individually fitted, hence the need for the ratio
    u_min: Max magnification of object, unitless
    t_0 is a variable we will not be fitting over, set it at zero by default'''

    #Adjusting to new units and unpacking theta
    tnew = t*24*(60**2)
    t_0new = t_0*24*(60**2)
    d_L = theta[0]*3.086e16
    d_S = theta[1]*3.086e16
    v_M_ratio = theta[2]*1e3/np.sqrt(1.99e30)
    u_min = theta[3]

    r_E = np.sqrt(4*(6.67e-11)*d_L*(d_S-d_L)/(9e16 * d_S)) #r_E without mass term
    us = np.sqrt(u_min**2 + (v_M_ratio*(tnew-t_0new)/r_E)**2)
    return get_mag(us)

def noisy_data_calc(low,upper,theta,noise,number,t_0=0):
    '''Generates a noisy lightcurve specified using the given parameters. Returns randomly selected t values and
    their corresponding magnification.
    low: Lower time bound of the dataset, in units of days. Must be negative
    upper: Upper time bound of the dataset, in units of days. Must be positive
    theta: Parameters of the lightcurve model - d_L, D_S, v_M_ratio, u_min
    noise: Standard deviation of the gaussian distribution from which noise is sampled
    number: Number of samples to be selected.
    t_0: Peak of lightcurve. Default 0'''

    ts = np.random.uniform(low,upper,number)
    mags = mean_function_theta(ts,theta,t_0) + np.random.normal(0,noise,number)
    return ts, mags
