###########################################
# Python implementation of WAIC computation taken from LaplacesDemon package
#############################################
import numpy as np
import numba as nb

@nb.jit(nopython=True)
def waic_calculation(x):
    """
    Numba-optimized function for WAIC comp
    """
    lppd = np.sum(np.log(np.mean(np.exp(x), axis=0)))
    
    # Compute the pointwise variance of the log-likelihood
    p_waic = np.sum(np.var(x, axis=0, ddof=1))
    
    # Compute WAIC
    waic = -2 * (lppd - p_waic)
    
    return waic