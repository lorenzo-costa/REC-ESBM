###########################################
# this file contains a bunch of numba optimised functions 
# to compute stuff
##############################################

import numba as nb
from math import lgamma
import numpy as np
from numba import cuda

###########################################
@nb.jit(nopython=True, fastmath=True)
def compute_log_likelihood(a, 
                           b, 
                           nh, 
                           nk, 
                           eps, 
                           mhk, 
                           user_clustering, 
                           item_clustering, 
                           dg_u, 
                           dg_i, 
                           dg_cl_u, 
                           dg_cl_i, 
                           degree_param_users=0.5, 
                           degree_param_items=0.5, 
                           degree_corrected=False):
    """Function to compute log-likelihood of current clustering

    To understand the parameters, see the detailed descriptions in
    results/text/lorenzocosta_thesis.pdf.
    
    Parameters
    ----------
    a : float
        Shape parameter for gamma prior
    b : float
        Rate parameter for gamma prior
    nh : array-like
        User cluster sizes
    nk : array-like
        Item cluster sizes
    eps : float
        Small value to avoid division by zero
    mhk : array-like
        mhk matrix
    user_clustering : array-like
        User clustering assignments
    item_clustering : array-like
        Item clustering assignments
    dg_u : array-like
        User degree sequence
    dg_i : array-like
        Item degree sequence
    dg_cl_u : array-like
        Sum of user degrees for users within each cluster
    dg_cl_i : array-like
        Sum of item degrees for items within each cluster
    degree_param_users : float, optional
        Degree-correction parameter for users, by default 0.5
    degree_param_items : float, optional
        Degree-correction parameter for items, by default 0.5
    degree_corrected : bool, optional
        Whether to apply degree correction, by default False

    Returns
    -------
    out : float
        log-likelihood value
    """
    
    out = 0.0
    if degree_corrected is True:
        for h in range(len(nh)):
            idx = np.where(user_clustering==h)[0]
            for i in idx:
                out += lgamma(degree_param_users+dg_u[i]+eps)
            out -= lgamma(nh[h]*degree_param_users+dg_cl_u[h]+eps)
            out += (dg_cl_u[h]*np.log(nh[h]))
            out += lgamma(nh[h]*degree_param_users)
            out -= (nh[h]*lgamma(degree_param_users))
            
        for k in range(len(nk)):
            idx = np.where(item_clustering==k)[0]
            for i in idx:
                out += lgamma(degree_param_items+dg_i[i]+eps)
            out -= lgamma(nk[k]*degree_param_items+dg_cl_i[k]+eps)
            out += (dg_cl_i[k]*np.log(nk[k]))
            out += lgamma(nk[k]*degree_param_items)
            out -= (nk[k]*lgamma(degree_param_items))
                                
    for h in range(len(nh)):
        for k in range(len(nk)):
            out += lgamma(mhk[h, k]+a+eps)-(mhk[h,k]+a)*np.log((nh[h]*nk[k]+b))
    return out


###################################
# gibbs-type prior sampling scheme
####################################

@nb.jit(nopython=True)
def sampling_scheme(V, H, frequencies, bar_h, scheme_type, scheme_param, sigma, gamma):
    """Probability of sampling each cluster (and a new one) under Gibbs-type priors.

    Parameters
    ----------
    V : int
        number of data points (nodes)
    H : int
        number of clusters
    frequencies : array-like
        cluster frequencies
    bar_h : int
        maximum number of clusters (for DM prior)
    scheme_type : str
        type of the prior distribution
    scheme_param : float
        additional parameter for Gibbs-type priors
    sigma : float
        sigma parameter for Gibbs-type priors
    gamma : float
        additional parameter for the GN model

    Returns
    -------
    probs : array-like
        probabilities of sampling each cluster and a new cluster

    """
    
    if scheme_type == 'DM':
        if H < bar_h:
            probs = np.zeros(len(frequencies)+1)
            for i in range(len(frequencies)):
                probs[i] = frequencies[i]-sigma
            probs[-1] = -sigma*(bar_h-H)
        else:
            probs = np.zeros(len(frequencies))
            for i in range(len(frequencies)):
                probs[i] = frequencies[i]-sigma

    if scheme_type == 'DP':
        probs = np.zeros(len(frequencies)+1)
        for i in range(len(frequencies)):
            probs[i] = frequencies[i]
        probs[-1] = scheme_param

    if scheme_type == 'PY':
        probs = np.zeros(len(frequencies)+1)
        for i in range(len(frequencies)):
            probs[i] = frequencies[i]-sigma
        probs[-1] = scheme_param+H*sigma

    if scheme_type == 'GN':
        probs = np.zeros(len(frequencies)+1)
        for i in range(len(frequencies)):
            probs[i] = (frequencies[i]+1)*(V-H+gamma)
        probs[-1] = H*(H-gamma)
    return probs


#################################
# log probability computation for gibbs sampling steps
#################################

# main function picking implementation
def compute_log_prob(probs, 
                     mhk_minus, 
                     frequencies_primary_minus, 
                     frequencies_secondary,
                     y_values, 
                     epsilon, 
                     a, 
                     b, 
                     max_clusters, 
                     is_user_mode, 
                     degree_corrected,
                     degree_param, 
                     degree_cluster_minus, 
                     degree_node, 
                     device):
    """Function to compute log probabilities for Gibbs sampling steps.
    
    Wrapper to choose between CPU and GPU implementations. To understand the parameters, 
    see the detailed descriptions in results/text/lorenzocosta_thesis.pdf.

    Parameters
    ----------
    probs : array-like
        Array of probabilities.
    mhk_minus : array-like
        mhk matrix with user/item removed.
    frequencies_primary_minus : array-like
        Frequencies of users/items with user/item removed.
    frequencies_secondary : array-like
        Frequencies of items/users (opposite of primary).
    y_values : array-like
        Array of yih or yuk values.
    epsilon : float
        Small value to avoid division by zero.
    a : float
        Shape parameter for gamma prior.
    b : float
        Rate parameter for gamma prior.
    max_clusters : int
        Current number of clusters.
    is_user_mode : bool
        Flag indicating user mode.
    degree_corrected : bool
        Flag indicating if degree correction is applied.
    degree_param : float
        Degree-correction parameter.
    degree_cluster_minus : array-like
        Sum of degrees in the cluster with user/item removed.
    degree_node : float
        Degree of selected node.
    device : str
        Device to run the computations on ('cpu' or 'gpu').

    Returns
    -------
    log_probs : array-like
        Computed log probabilities.
    """
    
    if device == 'gpu':
        try:
            return compute_prob_gpu_unif(probs, mhk_minus, frequencies_primary_minus, frequencies_secondary,
                                         y_values, epsilon, a, b, max_clusters, degree_corrected,
                                         degree_cluster_minus, degree_node, degree_param, is_user_mode)
        except Exception as e:
            print(f"Error using GPU implementation: {e}")
            return compute_prob_cpu_unif(probs, mhk_minus, frequencies_primary_minus, frequencies_secondary,
                          y_values, epsilon, a, b, max_clusters, degree_corrected,
                          degree_cluster_minus, degree_node, degree_param, is_user_mode)
    else:
        return compute_prob_cpu_unif(probs, mhk_minus, frequencies_primary_minus, frequencies_secondary,
                          y_values, epsilon, a, b, max_clusters, degree_corrected,
                          degree_cluster_minus, degree_node, degree_param, is_user_mode)
        
        

@nb.njit(fastmath=True, parallel=False)
def compute_prob_cpu_unif(probs, 
                          mhk_minus, 
                          frequencies_primary_minus, 
                          frequencies_secondary,
                          y_values, 
                          epsilon, 
                          a, 
                          b, 
                          max_clusters, 
                          degree_corrected,
                          degree_cluster_minus, 
                          degree_node, 
                          degree_param, 
                          is_user_mode):
    """
    CPU version of log probability computation for Gibbs sampling steps.
    """
    log_probs = np.zeros_like(probs)
    a_plus_epsilon = a + epsilon
    lgamma_a = lgamma(a)
    log_b = np.log(b)
    lgamma_a_log_b = -lgamma_a + a * log_b
    
    # Set indices based on mode
    primary_range = range(max_clusters)
    
    for i in primary_range:
        p_i = 0.0
        freq_i = frequencies_primary_minus[i]
        
        for j in range(len(frequencies_secondary)):
            # swap indices based on user/item mode
            if is_user_mode:
                h, k = i, j  
            else:
                k, h = i, j  
            
            mhk_val = mhk_minus[h, k]
            y_val = y_values[j]  
            
            mhk_plus_a = mhk_val + a_plus_epsilon
            mhk_plus_y_plus_a = mhk_val + y_val + a_plus_epsilon
            
            log_freq_prod1 = np.log(b + freq_i * frequencies_secondary[j])
            log_freq_prod2 = np.log(b + (freq_i + 1) * frequencies_secondary[j])
            
            p_i += (lgamma(mhk_plus_y_plus_a) - lgamma(mhk_plus_a) +
                   (mhk_plus_a - epsilon) * log_freq_prod1 -
                   (mhk_plus_y_plus_a - epsilon) * log_freq_prod2)
            
        log_probs[i] += p_i
        
        if degree_corrected is True:
            first = lgamma(frequencies_primary_minus[i]*degree_param + degree_cluster_minus[i])
            second = lgamma((frequencies_primary_minus[i]+1)*degree_param+degree_cluster_minus[i]+degree_node)
            
            third = lgamma((frequencies_primary_minus[i]+1)*degree_param)
            fourth = lgamma(frequencies_primary_minus[i]*degree_param)
            
            fifth = (degree_cluster_minus[i]+degree_node)*np.log(frequencies_primary_minus[i]+1)
            sixth = degree_cluster_minus[i]*np.log(frequencies_primary_minus[i])
            
            log_probs[i] += (first - second + third - fourth + fifth - sixth)
                        
    # Handle new cluster case
    if len(log_probs) > max_clusters:
        p_new = 0.0
        for j in range(len(frequencies_secondary)):
            y_val = y_values[j]
            p_new += (lgamma(y_val + a_plus_epsilon) + lgamma_a_log_b -
                     (y_val + a) * np.log(b + frequencies_secondary[j]))
                
        log_probs[max_clusters] += p_new
        if degree_corrected is True:
            log_probs[max_clusters] += (lgamma(degree_param)- lgamma(degree_param+degree_node))
    
    return log_probs


def compute_prob_gpu_unif(probs, 
                          mhk_minus, 
                          frequencies_primary_minus, 
                          frequencies_secondary,
                          y_values, 
                          epsilon, 
                          a, 
                          b, 
                          max_clusters, 
                          degree_corrected,
                          degree_cluster_minus, 
                          degree_node, 
                          degree_param, 
                          is_user_mode):
    """
    Function organizing GPU computation.
    """

    d_log_probs = cuda.to_device(np.zeros_like(probs))
    d_mhk_minus = cuda.to_device(mhk_minus)
    d_frequencies_primary_minus = cuda.to_device(frequencies_primary_minus)
    d_frequencies_secondary = cuda.to_device(frequencies_secondary)
    d_y_values = cuda.to_device(y_values)
    d_degree_cluster_minus = cuda.to_device(degree_cluster_minus)

    # not sure how to specify this part, for now leave 128
    threads_per_block = 128
    blocks_per_grid = (max_clusters + threads_per_block - 1) // threads_per_block

    compute_prob_kernel[blocks_per_grid, threads_per_block](d_log_probs, d_mhk_minus, d_frequencies_primary_minus, 
                                                            d_frequencies_secondary,d_y_values, epsilon, a, b, max_clusters, 
                                                            degree_corrected, d_degree_cluster_minus, degree_node, degree_param, 
                                                            is_user_mode)
    
    if len(probs) > max_clusters:
        compute_extra_prob_kernel[1, 1](d_log_probs, d_frequencies_secondary, d_y_values, epsilon, a, b, max_clusters, 
                                        degree_corrected, degree_node, degree_param)
        
    # Copy log_probs back to host for final calculations
    log_probs = d_log_probs.copy_to_host()

    return log_probs


@cuda.jit
def compute_prob_kernel(log_probs, 
                        mhk_minus, 
                        frequencies_primary_minus, 
                        frequencies_secondary,
                        y_values, 
                        epsilon, 
                        a, 
                        b, 
                        max_clusters, 
                        degree_corrected,
                        degree_cluster_minus, 
                        degree_node, 
                        degree_param, 
                        is_user_mode):
    """
    GPU kernel to compute sampling probabilities.
    """
    
    i = cuda.grid(1)

    if i < max_clusters:
        p_i = 0.0
        freq_i = frequencies_primary_minus[i]
        a_plus_epsilon = a + epsilon

        for j in range(len(frequencies_secondary)):
            if is_user_mode is True:
                h, k = i, j
            else:
                k, h = i, j
                
            mhk_val = mhk_minus[h, k]
            y_val = y_values[j]

            mhk_plus_a = mhk_val + a_plus_epsilon
            mhk_plus_y_plus_a = mhk_val + y_val + a_plus_epsilon

            log_freq_prod1 = np.log(b + freq_i * frequencies_secondary[j])
            log_freq_prod2 = np.log(b + (freq_i + 1) * frequencies_secondary[j])

            p_i += (cuda.libdevice.lgamma(mhk_plus_y_plus_a) - cuda.libdevice.lgamma(mhk_plus_a) +
                (mhk_plus_a - epsilon) * log_freq_prod1 -(mhk_plus_y_plus_a - epsilon) * log_freq_prod2)

        log_probs[i] += p_i
        
        if degree_corrected is True:
            first = cuda.libdevice.lgamma(frequencies_primary_minus[i]*degree_param + degree_cluster_minus[i])
            second = cuda.libdevice.lgamma((frequencies_primary_minus[i]+1)*degree_param+degree_cluster_minus[i]+degree_node)
            
            third = cuda.libdevice.lgamma((frequencies_primary_minus[i]+1)*degree_param)
            fourth = cuda.libdevice.lgamma(frequencies_primary_minus[i]*degree_param)
            
            fifth = (degree_cluster_minus[i]+degree_node)*np.log(frequencies_primary_minus[i]+1)
            sixth = degree_cluster_minus[i]*np.log(frequencies_primary_minus[i])
            
            log_probs[i] += (first - second + third - fourth + fifth - sixth)
            

@cuda.jit
def compute_extra_prob_kernel(log_probs, 
                              frequencies_secondary, 
                              y_values, 
                              epsilon, 
                              a, 
                              b, 
                              max_clusters, 
                              degree_corrected, 
                              degree_node, 
                              degree_param):
    """
    GPU kernel to compute the probability for a new cluster (H+1 case).
    """
    p_new = 0.0
    a_plus_epsilon = a + epsilon
    lgamma_a_log_b = - cuda.libdevice.lgamma(a) + a * np.log(b)

    for j in range(len(frequencies_secondary)):
        y_val = y_values[j]
        p_new += (cuda.libdevice.lgamma(y_val + a_plus_epsilon) + lgamma_a_log_b - 
               (y_val + a) * np.log(b + frequencies_secondary[j]))
        log_probs[max_clusters] += p_new
        
        if degree_corrected is True:
            log_probs[max_clusters] += (cuda.libdevice.lgamma(degree_param)-cuda.libdevice.lgamma(degree_param+degree_node))


@nb.jit(nopython=True)
def compute_log_probs_cov(probs, 
                          idx, 
                          cov_types, 
                          cov_nch, 
                          cov_values, 
                          nh, 
                          alpha_c, 
                          alpha_0):
    """Compute contribution of covariates to log probabilities.

    Parameters
    ----------
    probs : array-like
        array of probabilities
    idx : int
        index of the user/item being considered
    cov_types : array-like
        types of covariates ('cat' for categorical)
    cov_nch : array-like
        nch matrices for categorical covariates
    cov_values : array-like
        covariate values for each user/item
    nh : array-like
        cluster sizes
    alpha_c : array-like
        alpha_c parameters for categorical covariates
    alpha_0 : float
        sum of alpha_c parameters

    Returns
    -------
    log_probs : array-like
        log probabilities contribution from covariates
    """
    
    log_probs = np.zeros_like(probs)
    for i in nb.prange(len(cov_types)):
        if cov_types[i]=='cat':
            c = cov_values[i][idx]
            nch = cov_nch[i]
            for h in range(len(nh)):
                log_probs[h] += np.log(nch[c, h]+alpha_c[c])-np.log(nh[h]+alpha_0)
            log_probs[-1] += np.log(alpha_c[c])-np.log(alpha_0)
    
    return log_probs

@nb.jit(nopython=True, parallel=False)
def compute_co_clustering_matrix(mcmc_draws_users):
    """
    Compute co-clustering matrix from MCMC draws.
    """
    n_iters, num_users = mcmc_draws_users.shape
    
    co_clustering_matrix_users = np.zeros((num_users, num_users))
    for it in nb.prange(n_iters):
        for user_one in range(num_users):
            for user_two in range(num_users):
                if mcmc_draws_users[it, user_one] == mcmc_draws_users[it, user_two]:
                    co_clustering_matrix_users[user_one, user_two] += 1

    return co_clustering_matrix_users



        
     