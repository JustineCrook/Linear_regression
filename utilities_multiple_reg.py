import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import factor, sample
from numpyro.infer import MCMC, NUTS, init_to_median
import matplotlib.pyplot as plt
from jax import vmap
import corner
import seaborn as sns
import scipy.odr as odr
from joblib import Parallel, delayed
from tqdm import tqdm


from utilities_general import dict_samples_to_array, gen_synthetic_data_two_lines



###############################################################################################################################
## 2-LINE LINEAR REGRESSION


"""
WAY 2:

# Priors
scatter = numpyro.sample("sig", dist.Uniform(0, 5))
mu_gauss = numpyro.sample("mu_gauss", dist.Uniform(xmin,xmax))
w_gauss = numpyro.sample("w_gauss", dist.Uniform(0, xmax-xmin))
A1 = numpyro.sample("A1", dist.Uniform(0, 2)) # slope 1
A2 = numpyro.sample("A2", dist.Uniform(0, 2)) # slope 2
xtrans = numpyro.sample("xtrans", dist.Uniform(xmin, xmax))  # transition point
ytrans = numpyro.sample("ytrans", dist.Uniform(ymin, ymax))  # transition point

def broken_line(x):
    return lax.cond(
        x <= xtrans,
        lambda _: A1 * (x - xtrans) + ytrans,
        lambda _: A2 * (x - xtrans) + ytrans,
        operand=None  # no input to the lambdas
    )

"""

def run_two_line_linear_regression(x, y, xerr, yerr, true_data=None, show_plots=True):

    # Create the data dictionary
    kwargs = {
        'x': x,
        'y': y,
        'xerr': xerr,
        'yerr': yerr,
        }
    
    xmin = np.min(x)
    xmax = np.max(x)

    def model(x, y, xerr, yerr):


        # Priors
        scatter = numpyro.sample("sig", dist.Uniform(0, 5))
        mu_gauss = numpyro.sample("mu_gauss", dist.Uniform(xmin,xmax))
        w_gauss = numpyro.sample("w_gauss", dist.Uniform(0, xmax-xmin))
        xtrans = numpyro.sample("xtrans", dist.Uniform(xmin, xmax))  # transition point
        A1 = numpyro.sample("A1", dist.Uniform(0, 2)) # slope 1
        A2 = numpyro.sample("A2", dist.Uniform(0, 2)) # slope 2
        B1 = numpyro.sample("B1", dist.Uniform(-2, 2))  # y-intercept at x = 0
        # Compute the offset B2 such that the line is continuous at xtrans
        B2 = (A1 - A2) * xtrans + B1

        def broken_line(x):
            return lax.cond(
                x <= xtrans,
                lambda _: A1 * x + B1,
                lambda _: A2 * x + B2,
                operand=None  # no input to the lambdas
            )

        with numpyro.plate("data", len(x)):
            xt = numpyro.sample("xt", dist.Normal(mu_gauss, w_gauss))
        
        # Log of equation (13) of Bartlett and Desmond 2023, without the first term
        t1 = dist.Normal(x, xerr).log_prob(xt).sum() 
        y_pred = vmap(broken_line)(xt)
        t2 = dist.Normal(y, jnp.sqrt(yerr**2 + scatter**2)).log_prob(y_pred).sum()
        loglikelihood_det = t1 + t2


        loglikelihood = loglikelihood_det


        # Add the log-likelihood to the model
        numpyro.factor("ll", loglikelihood)


    # Set up the MCMC
    rng_key = jax.random.PRNGKey(42)
    kernel = NUTS(model, init_strategy=init_to_median(num_samples=1000))
    mcmc = MCMC(kernel, num_warmup=5000, num_samples=1000,progress_bar=True)

    # Run the MCMC
    mcmc.run(rng_key, **kwargs)
    if show_plots: mcmc.print_summary(exclude_deterministic=False)


    # Extract results
    samples = mcmc.get_samples()
    subset_keys = ["A1", "A2", "B1", "xtrans", "mu_gauss", "w_gauss", "sig"]
    samples_subset = {k: samples[k] for k in subset_keys}
    samples_arr, names = dict_samples_to_array(samples_subset)
    print(samples_arr.shape)


    
    param_means = []
    param_stds = []
    all_results = []
    params = ["slope1", "slope2", "offset1", "xtrans", "mu_true", "w_true", "sig"]
    if true_data is not None: true_vals = [true_data[p] for p in params]
    else: true_vals = [None]*len(params)
    for i, (p, v) in enumerate(zip(subset_keys, true_vals)):
        vals = samples[p]
        mean = np.mean(vals)
        std = np.std(vals)
        param_means.append(mean)
        param_stds.append(std)
        if true_data is not None: all_results.append((mean-v)/std)
        else:  all_results.append(np.nan)

    
    
    if show_plots:

        # Plot corner plot
        if true_data is not None: fig = corner.corner(samples_arr, labels=names, truths=true_vals)
        else: fig = corner.corner(samples_arr, labels=names)
        plt.show()
        #plt.savefig("corner.png")

        # Print results
        A1_mean = param_means[0]
        A2_mean = param_means[1]
        B1_mean = param_means[2]
        xtrans_mean = param_means[3]
        mu_mean = param_means[4]
        scatter_mean = param_means[6]
        B2_mean = (A1_mean - A2_mean) * xtrans_mean + B1_mean

        
        print(f"Estimated line 1: y = {A1_mean:.4f}x + {B1_mean:.4f}")
        print(f'Estimated line 2: y = {A2_mean:.4f}x + {B2_mean:.4f}')
        print(f'Estimated scatter (sigma): {scatter_mean:.4f}')
        
        if true_data is not None:
            true_slope1 = true_data["slope1"]
            true_slope2 = true_data["slope2"]
            true_xtrans = true_data["xtrans"] 
            true_offset1 = true_data["offset1"] 
            true_offset2 =  ( true_slope1 - true_slope2 ) * true_xtrans + true_offset1
            print(f'True line 1: y = {true_slope1:.4f}x + {true_offset1:.4f}')
            print(f'True line 2: y = {true_slope2:.4f}x + {true_offset2:.4f}')
            print(f'True scatter (sigma): {true_data["sig"]:.4f}')


        # Plot results
        plt.figure(figsize=(10, 6))
        plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=".")
        x_plot = np.linspace(xmin-1, xmax +1, 100)
        y1_plot = A1_mean * x_plot + B1_mean
        plt.plot(x_plot, y1_plot, 'r-', label=f'Fitted: y1 = {A1_mean:.4f}x + {B1_mean:.4f}')
        y2_plot = A2_mean * x_plot + B2_mean
        plt.plot(x_plot, y2_plot, 'r-', label=f'Fitted: y2 = {A2_mean:.4f}x + {B2_mean:.4f}')
        if true_data is not None:
            y1_plot = true_slope1 * x_plot + true_offset1
            plt.plot(x_plot, y1_plot, 'r-', label=f'True: y1 = {true_slope1:.4f}x + {true_offset1:.4f}')
            y2_plot = true_slope2 * x_plot + true_offset2
            plt.plot(x_plot, y2_plot, 'r-', label=f'True: y2 = {true_slope2:.4f}x + {true_offset2:.4f}')
        plt.legend()
        plt.show()
        #plt.savefig("result.png")


        # Plot result distributions 
        fig, axes = plt.subplots(1, len(params), figsize=(16, 4))
        for i, (p, v, mean, std) in enumerate(zip(subset_keys, true_vals, param_means, param_stds)):
            sns.histplot(samples[p], kde=True, ax=axes[i], label = f"mean={mean:.4f}, \nstd={std:.4f}")
            if true_data is not None: axes[i].axvline(v,  color='red', linestyle='--')
            axes[i].set_title(p)
            axes[i].legend(fontsize=6)
        plt.tight_layout()
        plt.show()
        #plt.savefig("result_distributions.png")

    return all_results, samples_arr




def run_mcmc_iteration_two_line(i, nx, show):
    
    x, xerr, y, yerr, true_data = gen_synthetic_data_two_lines(  nx=nx, seed=i, plot=False)
    all_results, samples_arr = run_two_line_linear_regression(x, y, xerr, yerr, true_data=true_data, show_plots=show)
    
    return all_results, samples_arr



def test_data_runner_two_line(nrepeats, nx):

    show = (nrepeats ==1)

    parallel_results = Parallel(n_jobs=-1)(
        delayed(run_mcmc_iteration_two_line)(i, nx, show=show)
        for i in tqdm(range(nrepeats))
    )

    all_results, all_samples = zip(*parallel_results)
    all_results = list(all_results) # shape: (n_runs, n_params)
    all_samples = list(all_samples) # shape: (n_runs, n_samples, n_params)
        

    all_results = np.array(all_results)
    params = ["slope1", "slope2", "offset1", "xtrans", "mu_true", "w_true", "sig"]
    n_params = len(params)

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    plt.title("Using the upper limits:")
    for i, param in enumerate(params):
        spreads = all_results[:, i]
        sns.histplot(spreads, kde=True, ax=axes[i])
        axes[i].axvline(0, color='red', linestyle='--')
        axes[i].set_title(f"Distribution of results for {param}")
        axes[i].set_xlabel("(Mean - true)/std")
        axes[i].set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Corner plot
    _, _, _, _, true_data = gen_synthetic_data_two_lines(  nx=nx, seed=i, plot=False)
    true_vals = [true_data[p] for p in params]
    samples_for_corner = np.vstack(all_samples)  # shape: (n_runs * n_samples, n_params)
    fig = corner.corner(samples_for_corner, labels=params, truths=true_vals)
    plt.show()

  





###############################################################################################################################


def odr_two_line(x, y, xerr, yerr):

    # Define the linear model for ODR
    def linear_func(beta, x):
        slope1, slope2, xtrans, ytrans = beta
        result = np.zeros_like(x)
        mask = x < xtrans
        result[mask] = slope1*(x[mask] - xtrans) + ytrans
        result[~mask] = slope2*(x[~mask] - xtrans) + ytrans
        return result

    # Create a model object
    model = odr.Model(linear_func)

    # Create a data object with errors
    data = odr.RealData(x, y, sx=xerr, sy= yerr)

    # Set up the ODR with an initial guess
    odr_instance = odr.ODR(data, model, beta0=[0.3, 1.4, 0, 5.0])

    # Run the regression
    output = odr_instance.run()

    # Get the fitted parameters
    slope1, slope2, xtrans, ytrans = output.beta
    slope1_err, slope2_err, xtrans_err, ytrans_err = output.sd_beta
    print(f"Fitted parameters:")
    print(f"  Slope1 = {slope1:.3f} ± {slope1_err:.3f}")
    print(f"  Slope2 = {slope2:.3f} ± {slope2_err:.3f}")
    intercept1 = -slope1*xtrans + ytrans
    intercept2 = -slope2*xtrans + ytrans
    print(f"  Intercept1 = {intercept1:.3f}")
    print(f"  Intercept2 = {intercept2:.3f}")

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=".")
    xmin = np.min(x)
    xmax = np.max(x)
    x_plot = np.linspace(xmin-1, xmax +1, 100)
    y1_plot = slope1 * x_plot + intercept1
    plt.plot(x_plot, y1_plot, 'r-', label=f'Fitted: y1 = {slope1:.4f}x + {intercept1:.4f}')
    y2_plot = slope2 * x_plot + intercept2
    plt.plot(x_plot, y2_plot, 'r-', label=f'Fitted: y2 = {slope2:.4f}x + {intercept2:.4f}')
    plt.legend()
    plt.show()

