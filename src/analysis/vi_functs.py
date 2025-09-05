#########################################################
# python implementation of functions in R package 'mcclust.ext'
########################################################

import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform

def VI(cls, cls_draw):
    """Compute Variation of Information (VI) between two clusterings.

    Parameters
    ----------
    cls : np.array
        Cluster assignments to compare against.
    cls_draw : np.array
        Cluster assignments drawn from the posterior.

    Returns
    -------
    output : np.ndarray
        Variation of Information (VI) between the two clusterings.
    """
    if cls.ndim == 1:
        cls = cls.reshape(1, -1)  # Convert vector to row matrix
    
    if cls_draw.ndim==1:
        cls_draw = cls_draw[np.newaxis, :]
        
    n = cls_draw.shape[1]
    M = cls_draw.shape[0]
    
    def VI_compute(c):
        f = 0
        for i in range(n):
            ind = (c == c[i])
            f += np.log2(np.sum(ind))
            
            for m in range(M):
                indm = (cls_draw[m, :] == cls_draw[m, i])
                ind_indm = ind & indm
                f += (np.log2(np.sum(indm)) - 2 * np.log2(np.sum(ind_indm))) / M
        
        return f / n
    
    output = np.apply_along_axis(VI_compute, 1, cls)
    return output


def VI_lb(clustering, psm):
    """Calculate the Variation of Information lower bound for a clustering.

    Parameters
    ----------
    clustering : np.array
        Cluster assignments to evaluate.
    psm : np.array
        Posterior similarity matrix.

    Returns
    -------
    float
        Variation of Information lower bound.
    """
    
    n = len(clustering)
    vi = 0
    
    # For a single clustering vector
    if clustering.ndim == 1:
        for i in range(n):
            # Create indicator for same cluster
            ind = (clustering == clustering[i]).astype(float)
            # Calculate VI component
            vi += (np.log2(np.sum(ind)) + np.log2(np.sum(psm[i,])) - 
                  2 * np.log2(np.sum(ind * psm[i,]))) / n
    # For a matrix of clusterings
    else:
        vi_values = np.zeros(clustering.shape[0])
        for k in range(clustering.shape[0]):
            for i in range(n):
                ind = (clustering[k] == clustering[k,i]).astype(float)
                vi_values[k] += (np.log2(np.sum(ind)) + np.log2(np.sum(psm[i,])) - 
                               2 * np.log2(np.sum(ind * psm[i,]))) / n
        vi = vi_values
    
    return vi

def minVI(psm, cls_draw=None, method="avg", max_k=None):
    """Compute the clustering minimising the VI following Wade and Ghahramani (2018).

    Parameters
    ----------
    psm : np.array
        Posterior similarity matrix.
    cls_draw : np.array, optional
        Cluster assignments drawn from the posterior, by default None
    method : str, optional
        Method to use for minimization, by default "avg"
    max_k : int, optional
        Maximum number of clusters to consider for greedy optimization, by default int(np.ceil(psm.shape[0] / 8))

    Returns
    -------
    dict
        Dictionary containing the best cluster assignments and their corresponding VI value.
    """
    
    results = {}
    
    if method == "avg" or method == "all":
        if max_k is None:
            max_k = int(np.ceil(psm.shape[0] / 8))
        
        distance_matrix = 1 - psm
        condensed_dist = squareform(distance_matrix)        
        Z = linkage(condensed_dist, method='average')        
        cls_avg = np.zeros((max_k, psm.shape[0]), dtype=int)
        VI_avg = np.zeros(max_k)
        
        for k in range(1, max_k + 1):
            cls_avg[k-1] = cut_tree(Z, n_clusters=k).flatten()
            VI_avg[k-1] = VI_lb(cls_avg[k-1], psm)
        
        val_avg = np.min(VI_avg)
        cl_avg = cls_avg[np.argmin(VI_avg)]
        
        if method == "avg":
            return {
                "cl": cl_avg,
                "value": val_avg,
                "method": "avg"
            }
        else:
            results["avg"] = {
                "cl": cl_avg,
                "value": val_avg
            }
    
    if method == "comp" or method == "all":
        if max_k is None:
            max_k = int(np.ceil(psm.shape[0] / 8))
        
        distance_matrix = 1 - psm
        condensed_dist = squareform(distance_matrix)
        
        Z = linkage(condensed_dist, method='complete')
        
        cls_comp = np.zeros((max_k, psm.shape[0]), dtype=int)
        VI_comp = np.zeros(max_k)
        
        for k in range(1, max_k + 1):
            cls_comp[k-1] = cut_tree(Z, n_clusters=k).flatten()
            VI_comp[k-1] = VI_lb(cls_comp[k-1], psm)
        
        val_comp = np.min(VI_comp)
        cl_comp = cls_comp[np.argmin(VI_comp)]
        
        if method == "comp":
            return {"cl": cl_comp, "value": val_comp, "method": "comp"}
        else:
            results["comp"] = {"cl": cl_comp, "value": val_comp}
    
    if method == "draws" or method == "all":
        n = psm.shape[0]
        
        # CHECK THIS FUNCTION, SEEMS EQUAL TO VI_lb. 
        # was defined separately also in original package
        def EVI_lb_local(c):
            f = 0
            for i in range(n):
                ind = (c == c[i]).astype(float)
                f += (np.log2(np.sum(ind)) + np.log2(np.sum(psm[i,])) - 
                      2 * np.log2(np.sum(ind * psm[i,]))) / n
            return f
        
        VI_draws = np.zeros(cls_draw.shape[0])
        for i in range(cls_draw.shape[0]):
            VI_draws[i] = EVI_lb_local(cls_draw[i])
        
        val_draws = np.min(VI_draws)
        cl_draw = cls_draw[np.argmin(VI_draws)].copy()
        
        if method == "draws":
            return {"cl": cl_draw, "value": val_draws, "method": "draws"}
        else:
            results["draws"] = {"cl": cl_draw, "value": val_draws}
    
    if method == "all":
        best_method = min(results, key=lambda x: results[x]["value"])
        return {"cl": results[best_method]["cl"], "value": results[best_method]["value"], "method": best_method, "all_results": results}
    
    return {"error": "Method not fully implemented yet"}

