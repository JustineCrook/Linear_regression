import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kstest

###############################################################################################################################
## HELPER METHODS

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




def plot_posterior_diagnostics(samples, truth):
    """
    samples: array of shape [n_runs, n_samples, n_params]
    truth:   array of shape [n_runs, n_params]
    """

    n_runs, n_samples, n_params = samples.shape

    # Preallocate diagnostics
    z_scores = np.zeros((n_runs, n_params))
    pit_vals = np.zeros((n_runs, n_params))

    # Compute metrics for each run + parameter
    for r in range(n_runs):
        for p in range(n_params):
            post = samples[r, :, p]
            t = truth[r, p]

            mu = np.mean(post)
            sigma = np.std(post)

            # Z-score: (mean - truth)/std
            z_scores[r, p] = (mu - t) / sigma

            # PIT value: posterior CDF evaluated at truth
            pit_vals[r, p] = np.mean(post <= t)

    # ---- Plotting ----
    fig, axes = plt.subplots(3, n_params, figsize=(4*n_params, 10))

    for p in range(n_params):

        # ----- Row 1: Standardised error histogram -----
        ax = axes[0, p]
        ax.hist(z_scores[:, p], bins=20, density=True, alpha=0.7)
        ax.set_title(f"Param {p+1}: (mean - truth)/std")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        # ----- Row 2: PIT histogram -----
        ax = axes[1, p]
        ax.hist(pit_vals[:, p], bins=20, range=(0, 1), density=True, alpha=0.7)
        ax.plot([0, 1], [1, 1], "k--", linewidth=1)
        ax.set_title(f"Param {p+1}: PIT histogram")
        ax.set_xlim(0, 1)

        # ----- Row 3: PIT CDF + envelope + KS test -----
        ax = axes[2, p]
        sorted_pit = np.sort(pit_vals[:, p])
        cdf = np.linspace(0, 1, n_runs)

        # KS test
        ks_stat, ks_p = kstest(pit_vals[:, p], "uniform")

        # 95% uniform envelope
        lo = np.maximum(0, cdf - 1.36/np.sqrt(n_runs))    # KS 95% envelope
        hi = np.minimum(1, cdf + 1.36/np.sqrt(n_runs))

        ax.plot(sorted_pit, cdf, label="Empirical CDF")
        ax.plot([0,1],[0,1],"k--",label="Ideal CDF")

        ax.fill_between(sorted_pit, lo, hi, color="gray", alpha=0.3,
                        label="95% KS envelope")

        ax.set_title(f"Param {p+1}: PIT CDF (KS={ks_stat:.3f}, p={ks_p:.3f})")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)

    fig.tight_layout()
    return fig





###############################################################################################################################
## GENERATE SYNTHETIC DATA



def convert_Fr(Fr_mJy, Fr_mJy_unc, d_kpc, d_kpc_unc, nu_GHz=1.28):
    """
    Convert radio flux density in units of mJy to radio luminosity in units of erg/s. Also returns error -- assuming nu has no uncertainty
    """

    S = Fr_mJy * 1e-3 * 1e-23         # convert mJy to Jy, then Jy to erg s^-1 cm^-2 Hz^-1
    d = d_kpc * 1e3 * 3.086e18       # convert kpc to cm
    nu = nu_GHz * 1e9                # convert GHz to Hz

    # Calculate luminosity
    L = 4 * np.pi * d**2 * S * nu    # in erg/s
    L_unc = L*np.sqrt( (Fr_mJy_unc/Fr_mJy)**2 + (2* d_kpc_unc/d_kpc)**2)

    return L, L_unc


################


def gen_synthetic_data(seed=0, return_alt=False, nx=None, verbose=True, no_uplims=False, change_line=False):

    # Set the seed for generating xtrue, xobs, ytrue, yobs values
    # Note that I also use the seed for generating xtrue and ytrue because we have a small number of data points, 
    # and if we don't, it will bias mu_gauss values across all repeats towards xtrue instead of of mu_true, 
    # and will bias w_gauss across all repeats towards the std of xtrue instead of w_true
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    #####################

    ## READ IN THE DATA
    path = "../DATA/interpolated_lrlx.txt"
    df = pd.read_csv(path, sep=',', header=0, encoding='utf-8')

    ## Get the BH HS/QS data, ignoring Lx upper limits
    mask = (df['class'].isin(["BH", "candidateBH"]) & (df['state'].isin(["HS", "QS"]))  & (df["Lx_uplim_bool"]==0) )
    filtered_df = df[mask] 
    D_kpc =  filtered_df['D'].to_numpy() # distance in kpc for each data point
    Lx = filtered_df['Lx'].to_numpy()
    Lx_unc_l = filtered_df['Lx_unc_l'].to_numpy()
    Lx_unc_u = filtered_df['Lx_unc_u'].to_numpy()
    Lx_uplims = filtered_df['Lx_uplim_bool'].astype(bool)
    Lr = filtered_df['Lr'].to_numpy()
    Lr_unc_l = filtered_df['Lr_unc'].to_numpy()     
    Lr_uplims = filtered_df['Lr_uplim_bool'].astype(bool)
    Lr_unc_u = np.copy(Lr_unc_l)
    Lr_unc_u[Lr_uplims] = 0
    
    
    ## Make a detection threshold of 3*rms where rms ~20uJy, using D_kpc
    rms = 20e-6 # Jy
    rms_erg_s, _ = convert_Fr(rms*1e3, 0, D_kpc, 0, nu_GHz=1.28) # erg/s
    threshold_Lr = 3 * np.array(rms_erg_s)

    #####################

    ## LINE PARAMETERS
    Lx0 = 5.21e35
    Lr0 = 5.14e28

    if change_line:
        beta = np.random.uniform(-2, 2) # slope
        norm_log = np.random.uniform(-2, 2) # intercept in log space
        sigma_eps_log = np.random.uniform(0.01, 2) # scatter in log space

        min_x = np.min(np.log10(Lx/Lx0))
        max_x = np.max(np.log10(Lx/Lx0))
        mu_true = np.random.uniform(1.1*min_x, 0.9*max_x) # mean of x distribution
        w_true = np.random.uniform(0, 0.9*(max_x - min_x)) # std of x distribution

    else:
        beta = 0.612
        norm_log = 0.12 # normalisation in log space, for linear fit
        sigma_eps_log = 0.5 # intrinsic scatter in log space, for linear fit
        
        xtrue_ref = np.log10(Lx/Lx0)
        mu_true = np.mean(xtrue_ref)
        w_true = np.std(xtrue_ref, ddof=1)

    # Normalisation in linear space, for powerlaw fit
    alpha = 10**norm_log # = (Lr0/(Lx0)**beta) * 10**(norm_log)

    if verbose:
        print("Line parameters:")
        print("beta (slope):", beta)
        print("norm_log (intercept):", norm_log)
        print("sigma_eps_log (scatter):", sigma_eps_log)
        print("mu_true (mean of x distribution):", mu_true)
        print("w_true (std of x distribution):", w_true)


    #####################

    ## MAKE SAMPLE DATA 
    # Generate sample data in log space directly

    ## Generate xtrue values 
    xerr = np.random.uniform(0.1/np.log(10), 0.5/np.log(10), size=xtrue_ref.shape)  # errors in log space are 1/log(10) * dx_lin/x_lin, so this corresponds to ~10-50% errors in linear space
    # Draw xtrue from a Gaussian with mean mu_true and standard deviation w_true
    xtrue = np.random.normal(mu_true, w_true, size=len(xtrue_ref)) 
    # Add the measurement error
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr

    ## Generate ytrue values, using the xtrue values generated above
    # Log space, i.e. for the linear fit
    ymean = norm_log + beta * xtrue # using the line parameters
    # Add intrinsic scatter
    ytrue = ymean + np.random.normal(size=len(ymean))* sigma_eps_log
    yerr = np.random.uniform(0.05/np.log(10), 0.4/np.log(10), size=xtrue.shape)   # errors in log space are 1/log(10) * dx_lin/x_lin, so this corresponds to ~5-40% errors in linear space
    # Add the measurement error
    yobs = ytrue + np.random.normal(size=len(xtrue)) * yerr
    yobs_orig = np.copy(yobs)


    ## Upper limits
    # Do this in linear space -- just easier / more intuitive in terms of understanding the <3*rms rule
    if no_uplims==False:
        
        ytrue_lin = 10**ytrue # = Lr / Lr0
        yobs_lin = 10**yobs # = Lr_obs / Lr0
        yerr_lin = ytrue_lin * np.log(10)* yerr  # = Lr_unc / Lr0 ; convert to linear space, just using small-ish error approximation... should be between 5-40%
        # Apply 3*rms rule for upper limits
        uplims_obs = (yobs_lin< threshold_Lr/Lr0) # mask
        yobs[uplims_obs] = np.log10(threshold_Lr[uplims_obs]/Lr0) # set the observed value to the threshold in log space
        
        if verbose: # Check 
            test = yerr_lin / ytrue_lin  *100   # should be between 5-40%
            plt.hist(test, bins=10)
            plt.xlabel("y-value % errors in linear space")
            plt.show()
    
    else: 
        uplims_obs = np.zeros(len(yobs), dtype=bool)


    ## Subsample if needed
    if nx is not None:
        indx = rng.choice(len(xobs), size=nx, replace=False)
        xtrue = xtrue[indx]
        xobs = xobs[indx]
        xerr = xerr[indx]
        ytrue = ytrue[indx]
        yobs = yobs[indx]
        yerr = yerr[indx]
        uplims_obs = uplims_obs[indx]
        yobs_orig = yobs_orig[indx]
        
    
    #####################

    ## PLOTTING

    ## Show the generated data
    if verbose:

        print("Number of data points in the txt file: ", len(Lx))
        print("Number of data points used: ", len(xtrue))
        print("Number of uplims: ", sum(uplims_obs))


        x_plot = np.logspace(np.log10(min(Lx/Lx0)), np.log10(max(Lx/Lx0)), 1000)
        x_plot_lin = np.log10(x_plot)

        
        #plt.errorbar(xobs, yobs, xerr=xerr, yerr=yerr, uplims = uplims_obs, fmt='o', label='Observed Data', color='blue')
        #plt.errorbar(xobs, yobs_orig, uplims = uplims_obs, fmt='o', label='Observed Data', color='green', ms=3)
        #x_plot_lin = np.log10(x_plot)
        #y_fit = np.log10(alpha) + beta * x_plot_lin
        #plt.plot( x_plot_lin  , y_fit, color='red' , label='True line')
        #plt.xlabel(r'$\log(L_X / L_{X0})$', fontsize=14)
        #plt.ylabel(r'$\log(L_R / L_{R0})$', fontsize=14)
        #plt.legend()
        #plt.show()
 

        print("True line: y = {:.4f}x + {:.4f}, sigma = {:.4f}".format(beta, norm_log, sigma_eps_log))

        ## Plot the true line
        x_plot_lin = np.log10(x_plot)
        y_fit = np.log10(alpha) + beta * x_plot_lin
        plt.plot( x_plot_lin  , y_fit, color='red' , label='True line')
        ## Plot the true observed data (if there are no detection limits)
        plt.errorbar(xobs, yobs_orig, uplims = uplims_obs, fmt='o', label='True Observed Data', color='green', ms=2)
        ## Plot the observed data, after accounting for detection limits
        plt.errorbar(xobs[~uplims_obs], yobs[~uplims_obs], xerr=xerr[~uplims_obs], yerr=yerr[~uplims_obs], fmt='o', label='Observed Data', color='blue')
        plt.errorbar(xobs[uplims_obs], yobs[uplims_obs], xerr=xerr[uplims_obs], yerr=yerr[uplims_obs], uplims= np.ones(sum(uplims_obs), dtype= bool), fmt='o', label='Upper Limits', color='orange')
        ## Plot settings
        plt.xlabel(r'$\log(L_X / L_{X0})$', fontsize=14)
        plt.ylabel(r'$\log(L_R / L_{R0})$', fontsize=14)
        plt.legend()
        plt.show()

        ## Histograms of true y values
        plt.hist(ytrue[~uplims_obs], alpha=0.5, label='ytrue detections', color='blue', density=True)
        plt.hist(ytrue[uplims_obs], alpha=0.5, label='ytrue uplims', color='orange', density=True)
        plt.xlabel("ytrue values")
        plt.legend()    
        plt.show()

        ## Histograms of true x values
        plt.hist(xtrue)
        plt.xlabel("xtrue values")
        plt.show()
        print("True mu (i.e. mean x_true):", mu_true)
        print("True w (i.e. std of x_true):", w_true)
        

    #####################

    ## PREPARE OUTPUTS

    ## Prepare array of the true values
    true_vals = [beta, norm_log, mu_true, w_true, sigma_eps_log] # [slope_true, offset_true, mu_true, w_true, sig_true]

    ## Separate detections and upper limits
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