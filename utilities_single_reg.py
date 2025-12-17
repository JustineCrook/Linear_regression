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

from utilities_general import *



params = ['A', 'B', 'mu_gauss', 'w_gauss', 'sig']
params_linmix = ["beta" , "alpha", "sig"]



###############################################################################################################################
## ADAPTED ROXY METHOD -- WITH UPLIM FUNCTIONALITY
# To do this, we leave xt as a free parameter for the upper limits


# i is the seed for generating the data
# nx is the number of detections
# nuplims is the number of upper limits
def run_linear_regression_with_uplims(i, nx=100, nuplims=20, plot=False, type_data = "new", no_uplims=False):
    
    if type_data == "old": 
        if nx is None: nx=100
        xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err, true_vals = gen_synthetic_data_old(nx, nuplims, seed=i, plot=False)
    elif type_data == "new": 
        xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err, true_vals = gen_synthetic_data(seed=i, nx=nx, plot=False, no_uplims=no_uplims)
    elif type_data == "change_line":
        xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err , true_vals = gen_synthetic_data_change_line(seed=i, nx=nx, plot=False, no_uplims=no_uplims)
    else:
        raise ValueError("type_data must be 'old', 'new' or 'change_line'")

    # Create the data dictionary
    kwargs = {
        'xdet': np.asarray(xdet),
        'ydet': np.asarray(ydet),
        'xdet_err': np.asarray(xdet_err),
        'ydet_err': np.asarray(ydet_err),
        'xuplim': np.asarray(xuplim),
        'yuplim': np.asarray(yuplim),
        'xuplim_err': np.asarray(xuplim_err),
        'yuplim_err': np.asarray(yuplim_err)
        }


    all_x = np.append(xdet, xuplim)
    min_x = np.min(all_x)
    max_x = np.max(all_x)


    def model(xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err):

        # Priors
        B = numpyro.sample("B", dist.Uniform(-5, 5)) # intercept
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
    # Same as:
    #param_means = np.mean(samples_arr, axis=0)
    #param_stds  = np.std(samples_arr, axis=0)
    #param_medians = np.median(samples_arr, axis=0)


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




###############################################################################################################################

## SAME FUNCTIONALITY AS ABOVE, BUT MARGANILISED OVER XT

def run_linear_regression_marginalised_xt(i, nx=None, nuplims=20, plot=False, type_data= "new", no_uplims=False):
    
    if type_data == "old": 
        if nx is None: nx=100
        xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err, true_vals = gen_synthetic_data_old(nx, nuplims, seed=i, plot=False)
    elif type_data == "new": 
        xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err, true_vals = gen_synthetic_data(seed=i, nx=nx, plot=False, no_uplims=no_uplims)
    elif type_data == "change_line":
        xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err , true_vals = gen_synthetic_data_change_line(seed=i, nx=nx,  plot=False, no_uplims=no_uplims)
    else:
        raise ValueError("type_data must be 'old', 'new' or 'change_line'")

    # Create the data dictionary
    kwargs = {
        'xdet': np.asarray(xdet),
        'ydet': np.asarray(ydet),
        'xdet_err': np.asarray(xdet_err),
        'ydet_err': np.asarray(ydet_err),
        'xuplim': np.asarray(xuplim),
        'yuplim': np.asarray(yuplim),
        'xuplim_err': np.asarray(xuplim_err),
        'yuplim_err': np.asarray(yuplim_err)
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

        loglikelihood = 0.0

        # DETECTIONS
        if len(xdet)>0:
            numerator_t1 = (w_gauss**2*(A*xdet + B - ydet)**2 + xdet_err**2*(A*mu_gauss + B - ydet)**2 + (ydet_err**2 + scatter**2)*(xdet - mu_gauss)**2) 
            denominator_t1 = (2*(A**2 * xdet_err**2 * w_gauss**2 + scatter**2*(xdet_err**2 + w_gauss**2) + (xdet_err**2 + w_gauss**2)*ydet_err**2))
            t2 = jnp.log(2 * jnp.pi * jnp.sqrt((A**2 * xdet_err**2 * w_gauss**2 + scatter**2 * (xdet_err**2 + w_gauss**2)+ (xdet_err**2 + w_gauss**2)*ydet_err**2)))
            loglikelihood = - jnp.sum(numerator_t1/denominator_t1 + t2)

        # UPPER LIMITS
        if len(xuplim)>0:
            t1 = dist.Normal(xuplim, jnp.sqrt(xuplim_err**2 + w_gauss**2)).log_prob(mu_gauss).sum()
            sigma_c_squared = (1/w_gauss**2 + 1/xuplim_err**2)**(-1)
            mu_c = sigma_c_squared * (mu_gauss/w_gauss**2 + xuplim/xuplim_err**2)
            sigma_squared = yuplim_err**2 + scatter**2 
            t2 =  jnp.sum( jnp.log( 0.5*erfc((A*mu_c + B - yuplim)/(jnp.sqrt( 2*sigma_squared + 2* A**2 *  sigma_c_squared ))) ))

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
        yplot_all = []
        for j in range(len(samples['A'])):
            A_sample = samples['A'][j]
            B_sample = samples['B'][j]
            y_plot = A_sample * x_plot + B_sample
            yplot_all.append(y_plot)
            if j<1000: plt.plot(x_plot, y_plot, 'r-', alpha=0.1) 
        # Plot best-fit
        y_plot = A_mean * x_plot + B_mean
        plt.plot(x_plot, y_plot, 'g-', label=f'Fitted: y = {A_median:.4f}x + {B_median:.4f}\nScatter: {scatter_median:.4f}')  
        # Plot 1-sigma band
        #y_mean = np.mean(yplot_all, axis=0)          # shape (100,)
        #y_std  = np.std(yplot_all, axis=0)           # shape (100,)
        #lower = y_mean - y_std
        #upper = y_mean + y_std
        #plt.fill_between(x_plot, lower, upper, alpha=0.8, color="green", label='1σ band (no scatter)')
        # Plot true fit
        y_plot = slope_true * x_plot + offset_true
        plt.plot(x_plot, y_plot, 'b-', label=f'True: y = {slope_true:.4f}x + {offset_true:.4f}\nScatter: {sig_true:.4f}') 
        plt.legend()
        plt.show()


        # Plot a corner plot
        fig = corner.corner(samples_arr, labels=params, truths=[slope_true, offset_true, mu_true, w_true, sig_true])
        plt.show()



    return normalised_results, samples_arr




###############################################################################################################################
## RUNNER FUNCTION


def runner_with_uplims(nrepeats, nx=100, nuplims=20, parallel=False, start_i=0,  plot=False, type_data="new", type_test="marg", no_uplims=False):
     
    #plot = ((nrepeats ==1) & (parallel==False))
    print("Show results of each iteration:", plot)

    if type_test == "marg": 
        func = run_linear_regression_marginalised_xt
        params_ = params
    elif type_test == "no_marg": 
        func = run_linear_regression_with_uplims
        params_ = params
    elif type_test == "linmix": 
        func = linmix_comparison
        params_ = params_linmix
    else: raise ValueError("type_test must be 'marg', 'no_marg' or 'linmix'")


    if type_data not in ["old", "new", "change_line"]: raise ValueError("type_data must be 'old', 'new' or 'change_line'")

    if parallel:
        parallel_results = Parallel(n_jobs=-1)(
            delayed(func)(start_i + i, nx, nuplims, type_data=type_data, plot=False,  no_uplims=no_uplims )
            for i in tqdm(range(nrepeats))
        )

        all_normalised_results, all_samples_arr = zip(*parallel_results)

    else:
        all_normalised_results = []
        all_samples_arr = []
        for i in range(nrepeats):
            print(f"RUN #{i}")

            normalised_results, samples_arr = func(start_i + i, nx, nuplims, type_data=type_data, plot=plot,  no_uplims=no_uplims )

            all_normalised_results.append(normalised_results)
            all_samples_arr.append(samples_arr)



    if type_data == "old": 
        _, _, _, _, _, _, _, _, true_vals = gen_synthetic_data_old(nx, nuplims, seed=0, plot=False)
        truth = np.array([list(true_vals)]*n_runs)
    elif type_data == "new": 
        _, _, _, _, _, _, _, _, true_vals = gen_synthetic_data(seed=0, nx=nx, plot=False)
        truth = np.array([list(true_vals)]*n_runs)
    elif type_data == "change_line": 
        truth = []
        for i in range(nrepeats):
            _, _, _, _, _, _, _, _, true_vals = gen_synthetic_data_change_line(seed=start_i+i, nx=nx, plot=False)
            truth.append(list(true_vals))
        truth = np.array(truth)
        print(truth)
    
    

    # Check plots
    all_samples_arr = np.asarray(all_samples_arr)
    n_runs, n_samples, n_params = all_samples_arr.shape
    print("n_runs:", n_runs, "n_samples:", n_samples, "n_params:", n_params)
    plot_posterior_diagnostics(all_samples_arr, truth)


    # Standardised results plot
    all_normalised_results = np.array(all_normalised_results) # shape: (n_runs, n_params)
    n_params = len(params_)
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


    # Corner plot
    if type_data != "change_line": # true values are the same for all runs

        slope_true, offset_true, mu_true, w_true, sig_true = true_vals 

        if type_test == "linmix":
            truths = [slope_true, offset_true, sig_true]
        else:
            truths = [slope_true, offset_true, mu_true, w_true, sig_true]

    
        # all_samples_arr has shape (n_runs, n_samples, n_params)
        samples_for_corner = np.vstack(all_samples_arr)  # stacks along the first axis, so resultant shape: (n_runs * n_samples, n_params)
        fig = corner.corner(samples_for_corner, labels=params_, truths=truths)
        plt.show()






###############################################################################################################################


def linmix_comparison(i, nx=100, nuplims=20, plot=False, previous_data_gen = False, no_uplims=False):
    """
    Compare the results of the linmix package with the results of this method.
    """
    
    if previous_data_gen: 
        if nx is None: nx=100
        xobs, xerr, yobs, yerr, delta, true_vals = gen_synthetic_data_old(nx, nuplims, seed=i, return_alt=True)
    else: xobs, xerr, yobs, yerr, delta, true_vals = gen_synthetic_data(seed=i, nx=nx, return_alt=True, no_uplims=no_uplims)


    np.random.seed(42)
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



    samples_subset = {}
    samples_subset['beta'] = np.asarray(betas)
    samples_subset['alpha'] = np.asarray(alphas)
    samples_subset['sig'] = np.asarray(sigmas)
    samples_arr, names = dict_samples_to_array(samples_subset)

    slope_true, offset_true, mu_true, w_true, sig_true = true_vals 

    # Get the required results
    param_means = []
    param_stds = []
    normalised_results = []
    param_medians =[]
    for i, (p, v) in enumerate(zip([ "beta","alpha", "sig"], [slope_true, offset_true, sig_true])):
        vals = samples_arr[:, i]
        mean = np.mean(vals)
        std = np.std(vals)
        median = np.median(vals)
        param_medians.append(median)
        param_means.append(mean)
        param_stds.append(std)
        normalised_results.append((mean-v)/std)



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

    return normalised_results, samples_arr



###############################################################################################################################