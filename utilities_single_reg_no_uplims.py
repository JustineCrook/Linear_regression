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


from utilities_single_reg import gen_synthetic_data, gen_synthetic_data_old, dict_samples_to_array

params = ['A', 'B', 'mu_gauss', 'w_gauss', 'sig']


###############################################################################################################################
## ROXY METHOD -- I.E. NO UPLIM FUNCTIONALITY


def run_linear_regression_without_uplims(x, y, xerr, yerr, true_vals=None, show_plots=False):

        
    if true_vals is None:
        true_vals = [None, None, None, None, None]

    # Create the data dictionary
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


    # Set up the MCMC
    rng_key = jax.random.PRNGKey(42)
    kernel = NUTS(model, init_strategy=init_to_median(num_samples=1000))
    mcmc = MCMC(kernel, num_warmup=5000, num_samples=1000,num_chains=1,progress_bar=True)

    # Run the MCMC
    mcmc.run(rng_key, **kwargs)
    if show_plots: mcmc.print_summary(exclude_deterministic=False)


    # Extract results
    #samples_per_chain = mcmc.get_samples(group_by_chain=True)
    samples = mcmc.get_samples()
    samples_subset = {k: samples[k] for k in params}
    samples_arr, names = dict_samples_to_array(samples_subset)
    print(samples_arr.shape)



    param_means = []
    param_stds = []
    results = []
    for i, (p, v) in enumerate(zip(params, true_vals)):
        vals = samples[p]# for all chains together
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        param_means.append(mean)
        param_stds.append(std)
        if v is not None: results.append((mean-v)/std)
  


    if show_plots:

        # Plot corner plot
        fig = corner.corner(samples_arr, labels=names, truths=true_vals)
        plt.show()
        plt.savefig("corner.png")

        # Print results
        A_mean = np.mean(param_means[0])
        B_mean = np.mean(param_means[1])
        scatter_mean = np.mean(param_means[4])
        print(f"Estimated line: y = {A_mean:.4f}x + {B_mean:.4f}")
        print(f"Estimated scatter (sigma): {scatter_mean:.4f}")

        if true_vals != [None]*5:
            slope_true, offset_true, mu_true, w_true, sig_true = true_vals
            print(f"True line: y = {slope_true:.4f}x + {offset_true:.4f}")
            print(f"True scatter (sigma): {sig_true:.4f}")


        # Plot results
        plt.figure(figsize=(10, 6))
        plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=".")
        x_plot = np.linspace(xmin-1, xmax +1, 100)
        y_plot = A_mean * x_plot + B_mean
        plt.plot(x_plot, y_plot, 'r-', label=f'Fitted: y = {A_mean:.4f}x + {B_mean:.4f}')
        plt.legend()
        plt.show()


        # Plot result distributions for all chains
        fig, axes = plt.subplots(1, len(params), figsize=(16, 4))
        for i, (p, v, mean, std) in enumerate(zip(params, true_vals, param_means, param_stds)):
            sns.histplot(samples[p], kde=True, ax=axes[i], label = f"mean={mean:.4f}, \nstd={std:.4f}")
            if v is not None: axes[i].axvline(v,  color='red', linestyle='--')
            axes[i].set_title(p)
            axes[i].legend(fontsize=6)
        plt.tight_layout()
        plt.show()


    return results



#############


def single_test_runner_without_uplims( own_data = None):
    
    if own_data is not None:
        xdet, ydet, xdet_err, ydet_err= own_data
    else: return 

    x = xdet
    y = ydet
    xerr = xdet_err
    yerr = ydet_err


    # Run the linear regression    
    results = run_linear_regression_without_uplims(x, y, xerr, yerr, show_plots=True)



#############


# If all_data = True, use both detections and upper limits as detections
# If all_data = False, use only detections
def runner_without_uplims(nrepeats, nx=100, nuplims=0, start_seed=0, previous_data_gen = False, all_data=True):    

    all_results = []  # shape: (n_runs, n_params)

    for i in range(nrepeats):

        print(f"RUN #{i}")

        # Generate the synthetic data, using the seed
        if previous_data_gen:
            xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err, true_vals = gen_synthetic_data_old(nx, nuplims, seed=start_seed+ i)
        else:
            xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err, true_vals = gen_synthetic_data(seed=start_seed+ i, nx=nx)
        slope_true, offset_true, mu_true, w_true, sig_true = true_vals 


        if all_data:
            x = np.concatenate([xdet, xuplim])
            y = np.concatenate([ydet, yuplim])
            xerr = np.concatenate([xdet_err, xuplim_err])
            yerr = np.concatenate([ydet_err, yuplim_err])
        else:
            x = xdet
            y = ydet
            xerr = xdet_err
            yerr = ydet_err



        # Run the linear regression    
        if nrepeats>1: show = False
        else: show = True
        results = run_linear_regression_without_uplims(x, y, xerr, yerr, true_vals, show_plots=show)
        all_results.append(results)


    if nrepeats>1: # Plot all results

        all_results = np.array(all_results)
        
        n_params = len(params)
        fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

        for i, param in enumerate(params):
            spreads = all_results[:, i]
            print(spreads)
            sns.histplot(spreads, kde=True, ax=axes[i])
            axes[i].axvline(0, color='red', linestyle='--')
            axes[i].set_title(f"Distribution of results for {param}")
            axes[i].set_xlabel("(Mean - true)/std")
            axes[i].set_ylabel("Count")

        plt.tight_layout()
        plt.show()



