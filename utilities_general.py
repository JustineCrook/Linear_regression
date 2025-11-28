import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################################################################
## HELPER METHOD

def dict_samples_to_array(samples):
    """Convert a dictionary of samples to a 2-dimensional array."""
    data = []
    names = []

    for key, x in samples.items():
        if x.ndim == 1:
            data.append(x)
            names.append(key)
        elif x.ndim == 2:
            for i in range(x.shape[-1]):
                data.append(x[:, i])
                names.append(f"{key}_{i}")
        elif x.ndim == 3:
            for i in range(x.shape[-1]):
                for j in range(x.shape[-2]):
                    data.append(x[:, j, i])
                    names.append(f"{key}_{j}_{i}")
        else:
            raise ValueError("Invalid dimensionality of samples to stack.")

    return np.vstack(data).T, names





###############################################################################################################################
## GENERATE SYNTHETIC DATA



def gen_synthetic_data_old(nx=100, nuplims=20, seed=0, plot=True, return_alt=False):


    ## Synthetic data parameters
    mu_true = -0.1    # Mean of the true x distribution
    w_true = 1.0     # Standard deviation of the true x distribution
    slope_true = 0.6
    offset_true = 3.0
    sig_true = 0.2 # additional scatter in the y direction
    x_err = 0.08 # x error for the data

    true_vals = [slope_true, offset_true, mu_true, w_true, sig_true] # ['A', 'B', 'mu_gauss', 'w_gauss', 'sig']

    # Set the seed for generating xtrue, xobs, ytrue, yobs values
    # Note that I also use the seed for generating xtrue and ytrue because we have a small number of data points, 
    # and if we don't, it will bias mu_gauss values across all repeats towards xtrue instead of of mu_true, 
    # and will bias w_gauss across all repeats towards the std of xtrue instead of w_true
    np.random.seed(seed)

    y_err =0.08

    # Generate xtrue values 
    # Draw xtrue from a Gaussian with mean mu_true and standard deviation w_true
    xtrue = np.random.normal(mu_true, w_true, size=nx) 

    # Generate xobs values
    # As a simple scheme, assume they all have the same error
    xobs = xtrue + np.random.normal(size=len(xtrue)) * x_err # equivalent to xtrue +  np.random.normal(loc=0.0, scale=x_err, size=len(xtrue))
    xerr = x_err*np.ones(len(xobs))

    # Generate ytrue values
    ymean = slope_true*xtrue + offset_true
    ytrue = ymean + np.random.normal(size=len(xtrue)) * sig_true

    # Generate the yobs values
    yobs = np.zeros(len(xtrue))
    # Calculate y
    yobs = ytrue + np.random.normal(size=len(xtrue)) * y_err
    yerr = y_err*np.ones(len(yobs))

    
    delta = np.ones(len(xobs), dtype=bool)  # 1 for detection, 0 for upper limit
    indices = np.random.choice(len(xobs), size=nuplims, replace=False)
    delta[indices] = False

    # Replace values for uplims
    yobs[~delta] = yobs[~delta] + 2* np.random.rand(len(xtrue[~delta])) # add uniform [0,2]

    # Extract values of interest
    xdet, ydet, xdet_err, ydet_err = xobs[delta], yobs[delta], xerr[delta], yerr[delta]
    xuplim, yuplim, xuplim_err, yuplim_err = xobs[~delta], yobs[~delta], xerr[~delta], yerr[~delta]


    if plot:

        plt.figure()
        plt.hist(xdet)
        print(np.mean(xdet))
        print(np.std(xdet, ddof=1))

        # Plot the generated data
        plt.figure()
        plt.errorbar(xdet, ydet, yerr=ydet_err, xerr=xdet_err, fmt=".")
        plt.errorbar(xuplim, yuplim, yerr=yuplim_err, xerr=xuplim_err, fmt="v")
        plt.tight_layout()
        plt.xlabel(r'$x_{\rm obs}$', fontsize=14)
        plt.ylabel(r'$y_{\rm obs}$', fontsize=14)
        plt.show()


    if return_alt: 
        delta = delta.astype(int)
        return xobs, xerr, yobs, yerr, delta, true_vals

    else: return xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err , true_vals



################


def gen_synthetic_data(seed=0, return_alt=False, nx=None):

    # Set the seed for generating xtrue, xobs, ytrue, yobs values
    # Note that I also use the seed for generating xtrue and ytrue because we have a small number of data points, 
    # and if we don't, it will bias mu_gauss values across all repeats towards xtrue instead of of mu_true, 
    # and will bias w_gauss across all repeats towards the std of xtrue instead of w_true
    np.random.seed(seed)

    rng = np.random.default_rng(seed)


    beta = 0.612
    Lx0 = 5.21e35
    Lr0 = 5.14e28
    norm_lin = 0.12 + np.log10(Lr0) - beta*np.log10(Lx0) # in log space, for linear fit
    sigma_eps_log = 0.5 # in log space, for linear fit

    # Normalisation in linear space, for powerlaw fit
    alpha = 10**norm_lin # = (Lr0/(Lx0)**beta) * 10**(0.12)

    path = "interpolated_lrlx.txt"
    # read as tab-separated, first line used as header (default header=0)
    df = pd.read_csv(path, sep=',', header=0, encoding='utf-8')


    mask = (df['class'].isin(["BH", "candidateBH"]) & (df['state'].isin(["HS", "QS"]))  & (df["Lx_uplim_bool"]==0) )
    filtered_df = df[mask] 

    Lx = filtered_df['Lx'].to_numpy()
    Lx_unc_l = filtered_df['Lx_unc_l'].to_numpy()
    Lx_unc_u = filtered_df['Lx_unc_u'].to_numpy()
    Lr = filtered_df['Lr'].to_numpy()
    Lr_unc_l = filtered_df['Lr_unc'].to_numpy()     
    Lr_uplims = filtered_df['Lr_uplim_bool'].astype(bool)
    Lr_unc_u = np.copy(Lr_unc_l)
    Lr_unc_u[Lr_uplims] = 0
    Lx_uplims = filtered_df['Lx_uplim_bool'].astype(bool)

    print("Number of uplims: ", sum(Lr_uplims))


    x_plot = np.logspace(np.log10(min(Lx)), np.log10(max(Lx)), 1000)
    x_plot_lin = np.log10(x_plot)


    ## MAKE SAMPLE DATA 

    # Generate sample data in log space directly
    xtrue = np.log10(Lx)
    xerr = np.maximum ( np.log10(Lx + Lx_unc_u) - np.log10(Lx), np.log10(Lx) - np.log10(Lx - Lx_unc_l) )
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr

    # Log space, i.e. for the linear fit
    ymean = norm_lin + beta * xtrue 
    ytrue = ymean + np.random.normal(size=len(ymean))* sigma_eps_log

    # Linear space -- just easier in terms of understanding the <3sigma rule
    ytrue_lin = 10**ytrue
    yerr_lin = np.random.uniform(0.05*ytrue_lin, 0.4*ytrue_lin, size=ytrue_lin.shape) 
    yobs_lin = ytrue_lin + np.random.normal(size=len(ytrue_lin)) * yerr_lin

    yobs_lin_orig = np.copy(yobs_lin)

    # Upper limits
    uplims_obs = (yobs_lin< (3*yerr_lin))
    yobs_lin[uplims_obs] = 3*yerr_lin[uplims_obs]

    # Convert back to log space
    yerr = np.maximum ( np.log10(ytrue_lin + yerr_lin) - np.log10(ytrue_lin), np.log10(ytrue_lin) - np.log10(ytrue_lin - yerr_lin) )
    yobs = np.log10(yobs_lin)


    if nx is not None:
        indx = rng.choice(len(xobs), size=nx, replace=False)
        xtrue = xtrue[indx]
        xobs = xobs[indx]
        xerr = xerr[indx]
        ytrue = ytrue[indx]
        yobs = yobs[indx]
        yerr = yerr[indx]
        uplims_obs = uplims_obs[indx]
        yobs_lin_orig = yobs_lin_orig[indx]
        

    print("Number of uplims: ", sum(uplims_obs))

    plt.errorbar(xobs, yobs, xerr=xerr, yerr=yerr, uplims = uplims_obs, fmt='o', label='Observed Data', color='blue')
    plt.errorbar(xobs, np.log10(yobs_lin_orig), uplims = uplims_obs, fmt='o', label='Observed Data', color='green', ms=3)
    x_plot_lin = np.log10(x_plot)
    y_fit = np.log10(alpha) + beta * x_plot_lin
    plt.plot( x_plot_lin  , y_fit, color='red' )
    plt.legend()
    plt.show()


    plt.hist(xtrue)
    plt.show()


    mu_true = np.mean(xtrue)
    w_true = np.std(xtrue, ddof=1)
    print("True mu:", mu_true)
    print("True w:", w_true)
    print("True line: y = {:.4f}x + {:.4f}, sigma = {:.4f}".format(beta, norm_lin, sigma_eps_log))


    true_vals = [beta, norm_lin, mu_true, w_true, sigma_eps_log] # [slope_true, offset_true, mu_true, w_true, sig_true]

    delta = np.ones(len(xobs), dtype=bool)  # 1 for detection, 0 for upper limit
    delta[uplims_obs] = False


    xdet = xobs[delta]  
    ydet = yobs[delta]
    xdet_err = xerr[delta]
    ydet_err = yerr[delta]
    xuplim = xobs[~delta]
    yuplim = yobs[~delta]
    xuplim_err = xerr[~delta]
    yuplim_err = yerr[~delta]

    if return_alt: 
        delta = delta.astype(int)
        return xobs, xerr, yobs, yerr , delta, true_vals

    else: return xdet, ydet, xdet_err, ydet_err, xuplim, yuplim, xuplim_err, yuplim_err , true_vals





###############################################################################################################################
## GENERATE SYNTHETIC DATA -- TWO LINES



def gen_synthetic_data_two_lines(  nx=100, seed=0, plot=True):


    data = {
    'mu_true': 0.4,
    'w_true': 1.6,
    'slope1': 0.2,
    'slope2': 0.7,
    'offset1': 0.09,
    'xtrans': 1.0,
    'xerr': 0.08,
    'yerr': 0.08,
    'sig': 0.1,
    }

    np.random.seed(seed)

    # mu_true is the mean of the true x distribution     
    # w_true is the standard deviation of the true x distribution

    mu_true = data["mu_true"]    
    w_true =data["w_true"]      
    slope1 = data["slope1"] 
    offset1 = data["offset1"] 
    slope2 = data["slope2"] 
    transition = data["xtrans"]
    xerr = data["xerr"] 
    yerr = data["yerr"] 
    sig = data["sig"] 

    offset2 = (slope1 - slope2) * transition + offset1



    # Gaussian distribution for x
    xtrue = np.random.normal(mu_true, w_true, size=nx)
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    ytrue = np.zeros(len(xtrue))
    yobs = np.zeros(len(xtrue))
    mask1 = xtrue<transition
    mask2 = xtrue>=transition

    # First line
    ytrue[mask1] = slope1*xtrue[mask1] + offset1
    yobs[mask1] = ytrue[mask1] + np.random.normal(size=sum(mask1)) * np.sqrt(yerr ** 2 + sig ** 2)

    # Seond line
    ytrue[mask2] = slope2*xtrue[mask2] + offset2
    yobs[mask2] = ytrue[mask2] + np.random.normal(size=sum(mask2)) * np.sqrt(yerr ** 2 + sig ** 2)


    if plot:
        plot_kwargs = {'fmt':'.', 'markersize':1, 'zorder':1,
                    'capsize':1, 'elinewidth':1.0, 'alpha':1}
        plt.errorbar(xobs, yobs, xerr=xerr, yerr=yerr, **plot_kwargs, color='k')
        plt.xlabel(r'$x_{\rm obs}$', fontsize=14)
        plt.ylabel(r'$y_{\rm obs}$', fontsize=14)
        plt.tight_layout()

    xerr= np.ones(len(xobs))*xerr
    yerr= np.ones(len(xobs))*yerr

    return xobs, xerr, yobs, yerr, data