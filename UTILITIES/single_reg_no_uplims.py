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
from joblib import Parallel, delayed

from general import *

params = ['A', 'B', 'mu_gauss', 'w_gauss', 'sig']


###############################################################################################################################
## ROXY METHOD -- I.E. NO UPLIM FUNCTIONALITY
# I use the equations from the Bartlett and Desmond 2023 paper here, where xtrue is marginalised over.
###############################################################################################################################


def run_linear_regression_without_uplims(x, y, xerr, yerr, true_vals=[None, None, None, None, None], verbose =False):


    ## Create the data dictionary
    kwargs = {
        'x': x,
        'y': y,
        'x_err': xerr,
        'y_err': yerr
        }

    xmin = np.min(x)
    xmax = np.max(x)

    def model(x, y, x_err, y_err):

        # Priors
        B = numpyro.sample("B", dist.Uniform(-20, 20)) # intercept
        A = numpyro.sample("A", dist.Uniform(-5,5)) # slope
        scatter = numpyro.sample("sig", dist.Uniform(0, 5))
        mu_gauss = numpyro.sample("mu_gauss", dist.Uniform(xmin,xmax))
        w_gauss = numpyro.sample("w_gauss", dist.Uniform(0, xmax-xmin))

        # DETECTIONS
        # Equation (14) of Bartlett and Desmond 2023
        numerator_t1 = (w_gauss**2*(A*x + B - y)**2 + x_err**2*(A*mu_gauss + B - y)**2 + (y_err**2 + scatter**2)*(x - mu_gauss)**2) 
        denominator_t1 = (2*(A**2 * x_err**2 * w_gauss**2 + scatter**2*(x_err**2 + w_gauss**2) + (x_err**2 + w_gauss**2)*y_err**2))
        t2 = jnp.log(2 * jnp.pi * jnp.sqrt((A**2 * x_err**2 * w_gauss**2 + scatter**2 * (x_err**2 + w_gauss**2)+ (x_err**2 + w_gauss**2)*y_err**2)))
        loglikelihood = - jnp.sum(numerator_t1/denominator_t1 + t2)


        # Add the log-likelihood to the model
        numpyro.factor("ll", loglikelihood)


    ## Set up the MCMC
    rng_key = jax.random.PRNGKey(42)
    kernel = NUTS(model, init_strategy=init_to_median(num_samples=1000))
    mcmc = MCMC(kernel, num_warmup=5000, num_samples=1000,num_chains=1,progress_bar=verbose)

    ## Run the MCMC
    mcmc.run(rng_key, **kwargs)
    if verbose: mcmc.print_summary(exclude_deterministic=False)


    ## Extract results
    #samples_per_chain = mcmc.get_samples(group_by_chain=True)
    samples = mcmc.get_samples()
    samples_subset = {k: samples[k] for k in params}
    samples_arr, names = dict_samples_to_array(samples_subset)
    #print(samples_arr.shape)

    param_means = []
    param_medians = []
    param_stds = []
    results = []
    for i, (p, v) in enumerate(zip(params, true_vals)): # for each param
        vals = samples[p] # for all chains together
        mean = np.mean(vals)
        median = np.median(vals)
        std = np.std(vals, ddof=1)
        param_medians.append(median)
        param_means.append(mean)
        param_stds.append(std)
        if v is not None: results.append((mean-v)/std)
  


    if verbose:

        ## Plot corner plot
        fig = corner.corner(samples_arr, labels=names, truths=true_vals)
        plt.show()

        ## Print results
        A_mean = param_means[0]
        B_mean = param_means[1]
        scatter_mean = param_means[4]
        print(f"Estimated line using mean: y = {A_mean:.4f}x + {B_mean:.4f}")
        print(f"Estimated scatter (sigma) using mean: {scatter_mean:.4f}")
        A_median = param_medians[0]
        B_median = param_medians[1]
        scatter_median = param_medians[4]
        print(f"Estimated line using median: y = {A_median:.4f}x + {B_median:.4f}")
        print(f"Estimated scatter (sigma) using median: {scatter_median:.4f}")

        ## Print true values 
        if true_vals != [None, None, None, None, None]:
            slope_true, offset_true, mu_true, w_true, sig_true = true_vals
            print()
            print(f"True line: y = {slope_true:.4f}x + {offset_true:.4f}")
            print(f"True scatter (sigma): {sig_true:.4f}")


        ## Plot results
        plt.figure(figsize=(10, 6))
        plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=".")
        x_plot = np.linspace(xmin-1, xmax +1, 100)
        y_plot = A_mean * x_plot + B_mean
        plt.plot(x_plot, y_plot, 'r-', label=f'Fitted: y = {A_median:.4f}x + {B_median:.4f}')
        plt.legend()
        plt.show()


        ## Plot resultant distributions for all chains
        fig, axes = plt.subplots(1, len(params), figsize=(16, 4))
        for i, (p, v, mean, std) in enumerate(zip(params, true_vals, param_means, param_stds)): # for each param
            sns.histplot(samples[p], kde=True, ax=axes[i], label = f"mean={mean:.4f}, \nstd={std:.4f}")
            if v is not None: axes[i].axvline(v,  color='red', linestyle='--')
            axes[i].set_title(p)
            axes[i].legend(fontsize=6)
        plt.tight_layout()
        plt.show()


    return results, samples_arr




###############################################################################################################################

def regression_without_uplims_helper(i, nx=100, start_seed=0, verbose=False):
    """
    Helper function for running a single regression without uplims.

    If all_data = True, use both detections and upper limits as detections. If all_data = False, use only detections.
    """

    # Generate the synthetic data, using the seed -- no_uplims=True
    xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err, true_vals = gen_synthetic_data(seed=start_seed+ i, nx=nx, verbose=False, no_uplims=True)
    slope_true, offset_true, mu_true, w_true, sig_true = true_vals 

    # Check that there are no uplims
    assert len(yuplim) == 0, "There are upper limits in the data, but no_uplims=True was specified."

    results, samples_arr = run_linear_regression_without_uplims(xdet, ydet, xdet_err, ydet_err, true_vals, verbose=verbose)

    return results, samples_arr





def runner_without_uplims(nrepeats, nx=100, start_seed=0,parallel=False):    
    """
    Run multiple linear regressions without upper limits, repeating nrepeats times, each time changing the seed.

    If all_data = True, use both detections and upper limits as detections. If all_data = False, use only detections.
    """

    verbose = ((nrepeats ==1) & (parallel==False))
    print("Show results of each iteration:", verbose)

    
    if parallel: # Run the regressions in parallel
        parallel_results = Parallel(n_jobs=-1)(
            delayed(regression_without_uplims_helper)(i, nx, start_seed)
            for i in tqdm(range(nrepeats))
        )
        all_normalised_results, all_samples_arr = zip(*parallel_results)

    else: # Run the regressions sequentially
        all_normalised_results = []
        all_samples_arr = []
        for i in range(nrepeats):
            print(f"RUN #{i}")
            normalised_results, samples_arr = regression_without_uplims_helper(i, nx, start_seed, verbose=verbose)
            all_normalised_results.append(normalised_results)
            all_samples_arr.append(samples_arr)


    all_samples_arr = np.asarray(all_samples_arr)
    n_runs, n_samples, n_params = all_samples_arr.shape
    print("n_runs:", n_runs, "n_samples:", n_samples, "n_params:", n_params)
    
    ## Get the true values (the seed does not matter here)
    _, _, _, _, _, _, _, _, true_vals = gen_synthetic_data(seed=0, verbose=False)
    slope_true, offset_true, mu_true, w_true, sig_true = true_vals 
    truth = np.array([list(true_vals)]*n_runs)

    ## Check plots
    plot_posterior_diagnostics(all_samples_arr, truth)


    ## Standardised results plot
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


    ## Corner plot
    # all_samples_arr has shape (n_runs, n_samples, n_params)
    samples_for_corner = np.vstack(all_samples_arr)  # stacks along the first axis, so resultant shape: (n_runs * n_samples, n_params)
    fig = corner.corner(samples_for_corner, labels=params, truths=[slope_true, offset_true, mu_true, w_true, sig_true])
    plt.show()



###############################################################################################################################