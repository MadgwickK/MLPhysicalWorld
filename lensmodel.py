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