import numpy as np
from sklearn.metrics import log_loss
from lensmodel import mean_function, mean_function_with_planet


parameter_bounds = {
    'v_T_sqrt_lens_mass_ratio': [0, 1000],
    'd_L':                      [0, 10000],
    'd_S':                      [0, 10000],
    'u_min':                    [0, 4]
}


def objective_function(data, mean_func, parameters):
    loss = log_loss(data[:, 1], mean_func(data[:, 0], parameters))
    return loss


def initial_sampling(data, mean_func, bounds):

