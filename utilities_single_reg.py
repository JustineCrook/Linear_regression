import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import ImproperUniform, constraints
from numpyro import factor, sample
from numpyro.infer import MCMC, NUTS, init_to_median
import matplotlib.pyplot as plt
from jax.scipy.special import erf
from jax.scipy.special import erfc
import arviz as az
import seaborn as sns
import numpyro.distributions as dist
import corner
from joblib import Parallel, delayed
from tqdm import tqdm
import linmix
import pandas as pd

from utilities_general import dict_samples_to_array, gen_synthetic_data, gen_synthetic_data_old



params = ['A', 'B', 'mu_gauss', 'w_gauss', 'sig']



###############################################################################################################################
## ADAPTED ROXY METHOD -- WITH UPLIM FUNCTIONALITY
# To do this, we leave xt as a free parameter for the upper limits


# i is the seed for generating the data
# nx is the number of detections
# nuplims is the number of upper limits
def run_linear_regression_with_uplims(i, nx=100, nuplims=20, plot=False, previous_data_gen = False):
    
    if previous_data_gen: xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err, true_vals = gen_synthetic_data_old(nx, nuplims, seed=i, plot=False)
    else: xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err, true_vals = gen_synthetic_data(seed=i, nx=nx, plot=False)


    # Create the data dictionary
    kwargs = {
        'xdet': xdet,
        'ydet': ydet,
        'xdet_err': xdet_err,
        'ydet_err': ydet_err,
        'xuplim': xuplim,
        'yuplim': yuplim,
        'xuplim_err': xuplim_err,
        'yuplim_err': yuplim_err
        }

    all_x = np.append(xdet, xuplim)
    min_x = np.min(all_x)
    max_x = np.max(all_x)


    def model(xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err):

        # Priors
        B = numpyro.sample("B", dist.Uniform(-20, 20)) # intercept
        A = numpyro.sample("A", dist.Uniform(-5,5)) # slope
        scatter = numpyro.sample("sig", dist.Uniform(0, 5))
        mu_gauss = numpyro.sample("mu_gauss", dist.Uniform(min_x, max_x))
        w_gauss = numpyro.sample("w_gauss", dist.Uniform(0, max_x - min_x))


        if len(xdet)>0:
            with numpyro.plate("det_data", len(xdet)):
                xt_det = numpyro.sample("xt_det", dist.Normal(mu_gauss, w_gauss))
        
        if len(xuplim)>0:
            with numpyro.plate("uplim_data", len(xuplim)):
                xt_uplim = numpyro.sample("xt_uplim", dist.Normal(mu_gauss, w_gauss))


        loglikelihood = 0.0

        # DETECTIONS
        # Log of equation (13) of Bartlett and Desmond 2023, without the first term
        if len(xdet)>0:
            t1 = dist.Normal(xdet, xdet_err).log_prob(xt_det).sum() 
            t2 = dist.Normal(ydet, jnp.sqrt(ydet_err**2 + scatter**2)).log_prob(A*xt_det + B).sum() 
            loglikelihood_det = t1 + t2

            loglikelihood = loglikelihood_det


        # UPPER LIMITS
        # Log of equation (13) of Bartlett and Desmond 2023, without the first term, but with the last term integrated over yuplim from -infty to y0 
        if len(xuplim)>0:
            t1 = dist.Normal(xuplim, xuplim_err).log_prob(xt_uplim).sum() 
            t2 = jnp.sum( jnp.log( 0.5*erfc((A*xt_uplim + B - yuplim)/(jnp.sqrt( 2*yuplim_err**2 + 2*scatter**2 ))) ))
            loglikelihood_uplim = t1 + t2
    
            loglikelihood += loglikelihood_uplim


        # Add the log-likelihood to the model
        numpyro.factor("ll", loglikelihood)


    # Set up the MCMC
    rng_key = jax.random.PRNGKey(42)
    kernel = NUTS(model, init_strategy=init_to_median(num_samples=1000))
    mcmc = MCMC(kernel, num_warmup=5000, num_samples=1000,progress_bar=True)

    # Run the MCMC
    mcmc.run(rng_key, **kwargs)
    if plot: mcmc.print_summary(exclude_deterministic=False)


    # Extract results
    samples = mcmc.get_samples()

    
    # Convert to format needed for the corner plot
    samples_subset = {k: samples[k] for k in params} # params is a list of parameter names
    # samples_arr is a 2D NumPy array of shape (N, D), where N is the total number of samples and D the number of parameters. Each column corresponds to one parameter.
    samples_arr, names = dict_samples_to_array(samples_subset)

    # Get the required results
    param_means = []
    param_stds = []
    normalised_results = []
    param_medians =[]
    for i, (p, v) in enumerate(zip(params, true_vals)):
        vals = samples[p]
        mean = np.mean(vals)
        std = np.std(vals)
        median = np.median(vals)
        param_medians.append(median)
        param_means.append(mean)
        param_stds.append(std)
        normalised_results.append((mean-v)/std)


    slope_true, offset_true, mu_true, w_true, sig_true = true_vals 


    if plot:

        # Plot histograms of the parameters
        # Create subplots in one row
        fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 4))
        for i, param in enumerate(params):
            ax = axes[i]
            ax.hist(samples[param], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(param)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')

        # Print results
        A_mean =param_means[0]
        B_mean = param_means[1]
        scatter_mean = param_means[4]
        print(f"Estimated line: y = {A_mean:.4f}x + {B_mean:.4f}")
        print(f"Estimated scatter (sigma): {scatter_mean:.4f}")
        print(f"True line: y = {slope_true:.4f}x + {offset_true:.4f}")
        print(f"True scatter (sigma): {sig_true:.4f}")
        print()

        # Print results
        A_median = param_medians[0]
        B_median = param_medians[1]
        scatter_median = param_medians[4]
        print(f"Estimated line: y = {A_median:.4f}x + {B_median:.4f}")
        print(f"Estimated scatter (sigma): {scatter_median:.4f}")
        print(f"True line: y = {slope_true:.4f}x + {offset_true:.4f}")
        print(f"True scatter (sigma): {sig_true:.4f}")


        # Plot some samples
        x_plot = np.linspace(min_x-1, max_x +1, 100)
        plt.figure(figsize=(10, 6))
        plt.errorbar(xdet, ydet, yerr=ydet_err, xerr=xdet_err, fmt=".")
        plt.errorbar(xuplim, yuplim, yerr=yuplim_err, xerr=xuplim_err, fmt="v")
        for j in range(1000):
            A_sample = samples['A'][j]
            B_sample = samples['B'][j]
            y_plot = A_sample * x_plot + B_sample
            plt.plot(x_plot, y_plot, 'r-', alpha=0.1) 
        # Plot best-fit
        y_plot = A_mean * x_plot + B_mean
        plt.plot(x_plot, y_plot, 'g-', label=f'Fitted: y = {A_median:.4f}x + {B_median:.4f}\nScatter: {scatter_median:.4f}')  
        plt.legend()
        plt.show()


        # Plot a corner plot
        fig = corner.corner(samples_arr, labels=params, truths=[slope_true, offset_true, mu_true, w_true, sig_true])
        plt.show()



    return normalised_results, samples_arr




def runner_with_uplims(nrepeats, nx=100, nuplims=20, parallel=False, start_i=0,  plot=False, previous_data_gen = False):
     
    plot = ((nrepeats ==1) & (parallel==False))
    print("Show results of each iteration:", plot)

    if parallel:
        parallel_results = Parallel(n_jobs=-1)(
            delayed(run_linear_regression_with_uplims)(start_i + i, nx, nuplims, previous_data_gen=previous_data_gen, plot=False )
            for i in tqdm(range(nrepeats))
        )

        all_normalised_results, all_samples_arr = zip(*parallel_results)

    else:
        all_normalised_results = []
        all_samples_arr = []
        for i in range(nrepeats):
            print(f"RUN #{i}")

            normalised_results, samples_arr = run_linear_regression_with_uplims(start_i + i, nx, nuplims, previous_data_gen=previous_data_gen, plot=plot)

            all_normalised_results.append(normalised_results)
            all_samples_arr.append(samples_arr)


    # Standardised results plot
    all_normalised_results = np.array(all_normalised_results) # shape: (n_runs, n_params)
    n_params = len(params)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    for i, param in enumerate(params):
        spreads = all_normalised_results[:, i] # all values across runs for the i-th parameter
        sns.histplot(spreads, kde=True, ax=axes[i])
        axes[i].axvline(0, color='red', linestyle='--')
        axes[i].set_title(f"Distribution of results for {param}")
        axes[i].set_xlabel("(Mean - true)/std")
        axes[i].set_ylabel("Count")
    plt.tight_layout()
    plt.show()


    if previous_data_gen: _, _, _, _, _, _, _, _, true_vals = gen_synthetic_data_old(nx, nuplims, seed=i, plot=False)
    else: _, _, _, _, _, _, _, _, true_vals = gen_synthetic_data(seed=i, plot=False)
    slope_true, offset_true, mu_true, w_true, sig_true = true_vals 

    # Corner plot
    # all_samples_arr has shape (n_runs, n_samples, n_params)
    samples_for_corner = np.vstack(all_samples_arr)  # stacks along the first axis, so resultant shape: (n_runs * n_samples, n_params)
    fig = corner.corner(samples_for_corner, labels=params, truths=[slope_true, offset_true, mu_true, w_true, sig_true])
    plt.show()







###############################################################################################################################


def linmix_comparison(i, nx=100, nuplims=20, plot=False, previous_data_gen = False):
    """
    Compare the results of the linmix package with the results of this method.
    """
    
    if previous_data_gen: xobs, xerr, yobs, yerr, delta, true_vals = gen_synthetic_data_old(nx, nuplims, seed=i, return_alt=True)
    else: xobs, xerr, yobs, yerr, delta, true_vals = gen_synthetic_data(seed=i, nx=nx, return_alt=True)


    np.random.seed(i)
    lm = linmix.LinMix(x=xobs, y=yobs, xsig=xerr, ysig=yerr, delta=delta, K=2, parallelize=False, seed=i)
    lm.run_mcmc(miniter=5000,maxiter=10000,silent=False)


    print("Number of chains: ", len(lm.chain))
 

    ## Extract the fitted parameters for each chain 
    alphas = []
    betas = []
    sigmas = []
    for i in range(len(lm.chain)):
        alphas.append(lm.chain[i]['alpha']) # alpha = y-intercept = c, i.e. log(Xi)
        betas.append(lm.chain[i]['beta']) # m, i.e. slope beta
        sigmas.append(np.sqrt(lm.chain[i]['sigsqr'])) # "spread" of data around best-fit


    mean_intercept = np.mean(alphas)
    print("Mean intercept: ", mean_intercept)
    mean_slope = np.mean(betas)
    print("Mean slope: ", mean_slope)
    mean_sigma = np.mean(sigmas)
    print("Mean spread: ", mean_sigma)
    print()
    median_intercept = np.median(alphas)
    print("Median intercept: ", median_intercept)
    median_slope = np.median(betas)
    print("Median slope: ", median_slope)
    median_sigma = np.median(sigmas)
    print("Median spread: ", median_sigma)


    if plot:

        delta = delta.astype(bool)
        min_x = np.min(xobs)
        max_x = np.max(xobs)

        # Plot histograms of the parameters
        params_linmix = ["beta", "alpha", "sigsqr"]
        param_arrays = [betas, alphas, sigmas] 
        fig, axes = plt.subplots(1, len(params_linmix), figsize=(4 * len(params_linmix), 4))
        # Loop through each parameter and plot its histogram
        for i, (param_name, samples_array) in enumerate(zip(params_linmix, param_arrays)):
            ax = axes[i]
            ax.hist(samples_array, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(param_name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        

        # Plot some samples
        x_plot = np.linspace(min_x-1, max_x +1, 100)
        plt.figure(figsize=(10, 6))
        plt.errorbar(xobs[delta], yobs[delta], yerr=yerr[delta], xerr=xerr[delta], fmt=".")
        plt.errorbar(xobs[~delta], yobs[~delta], yerr=yerr[~delta], xerr=xerr[~delta], fmt="v")
        for j in range(1000):
            A_sample = lm.chain[j]['beta']
            B_sample = lm.chain[j]['alpha']
            y_plot = A_sample * x_plot + B_sample
            plt.plot(x_plot, y_plot, 'r-', alpha=0.1)   
        # Plot best-fit
        y_plot = median_slope * x_plot + median_intercept
        plt.plot(x_plot, y_plot, 'g-', label=f'Fitted: y = {median_slope:.4f}x + {median_intercept:.4f} \nScatter: {median_sigma:.4f}')
        plt.legend()
        plt.show()


         # Plot some MORE samples
        x_plot = np.linspace(min_x-1, max_x +1, 100)
        plt.figure(figsize=(10, 6))
        plt.errorbar(xobs[delta], yobs[delta], yerr=yerr[delta], xerr=xerr[delta], fmt=".")
        plt.errorbar(xobs[~delta], yobs[~delta], yerr=yerr[~delta], xerr=xerr[~delta], fmt="v")
        #for j in range(len(lm.chain) - 1000, len(lm.chain)): # last 1000 samples
        for j in range(10000): 
            A_sample = lm.chain[j]['beta']
            B_sample = lm.chain[j]['alpha']
            y_plot = A_sample * x_plot + B_sample
            plt.plot(x_plot, y_plot, 'r-', alpha=0.1)   
        # Plot best-fit
        y_plot = median_slope * x_plot + median_intercept
        plt.plot(x_plot, y_plot, 'g-', label=f'Fitted: y = {median_slope:.4f}x + {median_intercept:.4f} \nScatter: {median_sigma:.4f}')
        plt.legend()
        plt.show()



###############################################################################################################################