import numpy as np
from lensmodel import mean_function_theta
import emcee
import corner
import gaussian_process as GP
import matplotlib.pyplot as plt

def gaussian_process_update(theta,x_train, y_train, x_test, kernel_func, mean_func, noise_variance):
    '''Updates the mean and covariance functions of a Gaussian Process.
    
    Parameters:
        x_train (ndarray): Training input data 
        y_train (ndarray): Training output data
        x_test (ndarray): Test input data for predictions.
        kernel_func (function): Covariance function kernel 
        mean_func (function): Initial mean function 
        noise_variance (float): Noise variance parameter 
        theta (ndarray): List of parameters for the mean function
        
    Returns:
        mean_updated (ndarray): Updated mean function evaluated at x_test
        covariance_updated (ndarray): Updated covariance function evaluated at x_test'''

    # Number of training points
    n = x_train.shape[0]
    
    # Compute the kernel (covariance) matrices
    K = kernel_func(x_train, x_train,sigma_f = 1,sigma_l = 2) + noise_variance * np.eye(n)  # Add noise to the diagonal
    K_s = kernel_func(x_train, x_test,sigma_f = 1,sigma_l = 2)  # Cross-covariance between training and test
    K_ss = kernel_func(x_test, x_test,sigma_f = 1,sigma_l = 2)  # Covariance of test points
    
    # Compute the mean vector at test points
    K_inv = np.linalg.inv(K)
    mean_train = mean_func(x_train,theta)
    mean_test = mean_func(x_test,theta)
    
    mu = mean_test + K_s.T @ K_inv @ (y_train - mean_train)
    
    # Compute the covariance matrix at test points
    cov = K_ss - K_s.T @ K_inv @ K_s
    
    return mu, cov

def mcmc_fit(time,magnifications,bounds,initial,numiter,noise):
    '''Runs an MCMC sampler to optimise the parameters. Defines likelihood functions inside.
    
    Parameters:
        time (ndarray): Input array of time data
        magnifications (ndarray): Input array of magnification data
        bounds : Upper and lower bounds for parameters
        initial (ndarray): Best guess initial parameters
        numiter (int): Number of iterations for the MCMC
        noise (float): Noise added to Gaussian Process matrix
        
    Returns:
        sampler (class): emcee sampler chain'''

    def log_likelihood(magnifications,pred_mag,pred_cov):
        term1 = -len(magnifications)/2 * np.log(2*np.pi)
        term2 = -0.5 *len(magnifications) * np.sum(np.log(pred_cov**2))
        term3 = -0.5 * np.sum(((magnifications-pred_mag)/pred_cov)**2)
        return term1+term2+term3

    def log_prior(theta):
        t_E, t_E_bounds = theta[0], bounds['t_E']
        u_0, u_0_bounds = theta[1], bounds['u_min']
        t_0, t_0_bounds = theta[2], bounds['t_0']
        if t_E_bounds[0] < t_E < t_E_bounds[1] and t_0_bounds[0] < t_0 < t_0_bounds[1] and u_0_bounds[0] < u_0 < u_0_bounds[1]:
            return 0.0
        else:
            return -np.inf
        
    def log_probability(theta,time,magnifications):
        
        mu, cov = gaussian_process_update(theta,time.reshape(-1,1),magnifications.reshape(-1,1),time.reshape(-1,1),GP.matern52_kernel,mean_function_theta,noise)
        #Matern kernel works well for this, have tried with rbf kernel as well

        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + log_likelihood(magnifications,mu.flatten(),np.sqrt(np.diag(cov)))
    
    pos = initial + (1e-4 * np.random.randn(32, 3))
    nwalkers, ndim = pos.shape
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(time,magnifications))

    state = sampler.run_mcmc(pos, 100)
    sampler.reset() #Burn-in phase

    sampler.run_mcmc(state, numiter, progress=True)
    
    return sampler
    
def mcmc_results(sampler,burnin = 0):
    '''Takes the emcee sampler and produces results from the data. Produces chain and corner plots.
    
    Parameters:
        sampler (class): emcee sampler class from the mcmc_fit function
        
    Returns:
        theta_optimised (ndarray): Optimised parameters and their lower/upper bounds'''
    flat_samples = sampler.get_chain(discard=burnin,flat=True)
    samples = sampler.get_chain(discard=burnin)

    params = np.percentile(flat_samples, [16, 50, 84],axis=0)

    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    labels = [r"$t_E$", r"$u_{\mathrm{min}}$", r"$t_0$"]
    for i in range(3):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.show()

    fig = corner.corner(flat_samples, labels=labels, truths=params[1])   
    plt.show()

    return params

'''
#Uncomment the following code to run the MCMC. Alter parameters as needed
truevalues = [70,0.2,0]
parameter_bounds = {
    't_E':      [0.01, 500],    # days 
    't_0':      [-20, 20],        # days
    'u_min':    [0, 1]          # unitless
}

times, mags = noisy_data_calc(-100,100,truevalues,0.1,101)

#Instead of starting off at the true values, use the best initial guess. 
#There should be >50* the autocorrelation time samples. 5000 seems to work well. 
#Check autocorrelation time with print(sampler.get_autocorr_time())
sampler = mcmc_fit(times,mags,parameter_bounds,truevalues,5000,1)

print(mcmc_results(sampler))
'''
