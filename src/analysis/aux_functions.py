#####################
# misc functions 
#################
import numba as nb
import scipy as sc
import numpy as np


def compute_precision(true, preds):
    try:
        return len(set(true).intersection(set(preds)))/len(preds)
    except ZeroDivisionError:
        return 0.0

def compute_recall(true, preds):
    try:
        return len(set(true).intersection(set(preds)))/len(true)
    except ZeroDivisionError:
        return 0.0

# some fast functions
@nb.jit(nopython=True)
def gammaln_nb(x:int)->float:
    return sc.gammaln(x)

@nb.jit(nopython=True)
def rising_factorial(a, n):
    if n == 0:
        return 1
    return a*rising_factorial(a+1, n-1)

@nb.jit(nopython=True)
def truncated_factorial(n, k):
    result = 1
    for i in range(k):
        result *= (n-i)
    return result

def relabel_clusters(cluster_indices):
    unique_labels, counts = np.unique(cluster_indices, return_counts=True) 
    sorted_labels = unique_labels[np.argsort(-counts)] 
    
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}  # Create mapping
    relabeled_clusters = np.vectorize(label_mapping.get)(cluster_indices)  # Apply mapping

    return relabeled_clusters


################################################
# auxialiaries for plotting

# function to plot the heatmap representation

def plot_heatmap(model, user_covariates=None, item_covariates=None, add_labels=True, size=(12,10), capped=None):

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    
    if user_covariates is not None:
        user_covariates = user_covariates.copy()
    if item_covariates is not None:
        item_covariates = item_covariates.copy()

    user_clustering = model.user_clustering.copy()
    item_clustering = model.item_clustering.copy()
    Y = model.Y
    if capped is not None:
        Y = np.clip(Y, 0, capped)

    # Sort clusters by size
    user_cluster_sizes = {}
    for cluster in user_clustering:
        user_cluster_sizes[cluster] = user_cluster_sizes.get(cluster, 0) + 1

    item_cluster_sizes = {}
    for cluster in item_clustering:
        item_cluster_sizes[cluster] = item_cluster_sizes.get(cluster, 0) + 1

    sorted_user_clusters = sorted(user_cluster_sizes.keys(),
                                key=lambda x: user_cluster_sizes[x],
                                reverse=True)
    sorted_item_clusters = sorted(item_cluster_sizes.keys(),
                                key=lambda x: item_cluster_sizes[x],
                                reverse=True)

    user_cluster_rank = {cluster: i for i, cluster in enumerate(sorted_user_clusters)}
    item_cluster_rank = {cluster: i for i, cluster in enumerate(sorted_item_clusters)}

    idx_sort_users = sorted(np.arange(Y.shape[0]),
                        key=lambda i: user_cluster_rank[user_clustering[i]])
    idx_sort_items = sorted(np.arange(Y.shape[1]),
                        key=lambda i: item_cluster_rank[item_clustering[i]])

    sorted_user_clusters_list = [user_clustering[i] for i in idx_sort_users]
    sorted_item_clusters_list = [item_clustering[i] for i in idx_sort_items]

    user_cluster_boundaries = [0]
    prev_cluster = sorted_user_clusters_list[0]
    for i, cluster in enumerate(sorted_user_clusters_list[1:], 1):
        if cluster != prev_cluster:
            user_cluster_boundaries.append(i)
            prev_cluster = cluster
    user_cluster_boundaries.append(len(sorted_user_clusters_list))  # final bound

    item_cluster_boundaries = [0]
    prev_cluster = sorted_item_clusters_list[0]
    for i, cluster in enumerate(sorted_item_clusters_list[1:], 1):
        if cluster != prev_cluster:
            item_cluster_boundaries.append(i)
            prev_cluster = cluster
    item_cluster_boundaries.append(len(sorted_item_clusters_list))  # final bound

    # Create figure with appropriate size and layout for covariates
    fig = plt.figure(figsize=size)

    # Set the layout based on whether we have covariates
    if user_covariates is not None or item_covariates is not None:
        # Define grid for main heatmap and covariate bars
        width_ratios = [0.02, 1] if user_covariates is not None else [0, 1]
        height_ratios = [1, 0.02] if item_covariates is not None else [1, 0]
        gs = fig.add_gridspec(2, 2, width_ratios=width_ratios, 
                                height_ratios=height_ratios,
                                wspace=0.01, hspace=0.01)
        
        # Main heatmap
        ax_heatmap = fig.add_subplot(gs[0, 1])
        
        # User covariates on the left
        if user_covariates is not None:
            ax_user_cov = fig.add_subplot(gs[0, 0], sharey=ax_heatmap)
        
        # Item covariates on the bottom
        if item_covariates is not None:
            ax_item_cov = fig.add_subplot(gs[1, 1], sharex=ax_heatmap)
    else:
        ax_heatmap = fig.add_subplot(111)

    # Plot main heatmap
    heatmap = sns.heatmap(Y[idx_sort_users, :][:, idx_sort_items], ax=ax_heatmap, cbar_kws={"shrink": 0.8})

    # Add cluster boundaries
    for boundary in user_cluster_boundaries:
        ax_heatmap.axhline(y=boundary, color='white', linewidth=2)
    for boundary in item_cluster_boundaries:
        ax_heatmap.axvline(x=boundary, color='white', linewidth=2)

    # Add titles and labels
    # ax_heatmap.set_title('Heatmap with Largest Clusters in Top-Left Corner', fontsize=15, pad=20)
    ax_heatmap.set_xlabel('Items', fontsize=14, labelpad=20)
    ax_heatmap.set_ylabel('Users', fontsize=14, labelpad=20)

    # Add cluster labels if requested
    if add_labels:
        for i in range(len(user_cluster_boundaries) - 1):
            cluster_label = sorted_user_clusters[i]
            mid_point = (user_cluster_boundaries[i] + user_cluster_boundaries[i+1]) / 2
            ax_heatmap.text(-0.5, mid_point, f'C{cluster_label}',
                    verticalalignment='center', horizontalalignment='right',
                    fontsize=12)
        
        for i in range(len(item_cluster_boundaries) - 1):
            cluster_label = sorted_item_clusters[i]
            mid_point = (item_cluster_boundaries[i] + item_cluster_boundaries[i+1]) / 2
            ax_heatmap.text(mid_point, Y.shape[0] + 0.5, f'C{cluster_label}',
                    verticalalignment='top', horizontalalignment='center',
                    fontsize=12)

    ax_heatmap.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False, left=False, right=False, labelleft=False)

    # Process and plot user covariates if provided
    if user_covariates is not None:
        # Extract covariates for sorted users
        sorted_user_covs = [user_covariates[i] for i in idx_sort_users]
        
        # Get unique categories and assign colors
        unique_user_cats = sorted(set(sorted_user_covs))
        user_cmap = plt.get_cmap('tab10', len(unique_user_cats))
        user_color_dict = {cat: user_cmap(i) for i, cat in enumerate(unique_user_cats)}
        
        # Create color matrix for user covariates
        n_users = len(sorted_user_covs)
        user_cov_matrix = np.zeros((n_users, 1))
        for i, cov in enumerate(sorted_user_covs):
            # Convert category to numeric index for colormapping
            user_cov_matrix[i, 0] = unique_user_cats.index(cov)
        
        # Plot user covariates with proper alignment
        # The extent parameter is set to match the axis limits of the heatmap
        user_img = ax_user_cov.imshow(user_cov_matrix, aspect='auto', origin='upper', 
                                    cmap=user_cmap, extent=[0, 1, n_users, 0])
        
        ax_user_cov.set_xticks([])
        ax_user_cov.set_yticks([])
        
        # Add a legend for user covariates
        user_legend_elements = [Rectangle((0, 0), 1, 1, color=user_color_dict[cat], label=str(cat)) 
                                for cat in unique_user_cats]
        ax_user_cov.legend(handles=user_legend_elements, title='User Covariates',
                        bbox_to_anchor=(0.1, 0.1), loc='center right')

    # Process and plot item covariates if provided
    if item_covariates is not None:
        # Extract covariates for sorted items
        if isinstance(item_covariates, dict):
            sorted_item_covs = [item_covariates[i] for i in idx_sort_items]
        else:  # Assume it's a list
            sorted_item_covs = [item_covariates[i] for i in idx_sort_items]
        
        # Get unique categories and assign colors
        unique_item_cats = sorted(set(sorted_item_covs))
        item_cmap = plt.get_cmap('tab10', len(unique_item_cats))
        item_color_dict = {cat: item_cmap(i) for i, cat in enumerate(unique_item_cats)}
        
        # Create color matrix for item covariates
        n_items = len(sorted_item_covs)
        item_cov_matrix = np.zeros((1, n_items))
        for i, cov in enumerate(sorted_item_covs):
            # Convert category to numeric index for colormapping
            item_cov_matrix[0, i] = unique_item_cats.index(cov)
        
        # Plot item covariates with proper alignment
        # The extent parameter is set to match the axis limits of the heatmap
        # Align the width of ax_item_cov with ax_heatmap
        heatmap_pos = ax_heatmap.get_position()
        ax_item_cov.set_position([heatmap_pos.x0, ax_item_cov.get_position().y0, 
                                heatmap_pos.width, ax_item_cov.get_position().height])

        # Plot item covariates ensuring correct width
        item_img = ax_item_cov.imshow(item_cov_matrix, aspect='auto', origin='upper', cmap=item_cmap, 
                                    extent=[0, n_items, 1, 0])

        
        ax_item_cov.set_xticks([])
        ax_item_cov.set_yticks([])
        
        # Add a legend for item covariates
        item_legend_elements = [Rectangle((0, 0), 1, 1, color=item_color_dict[cat], label=str(cat)) 
                                for cat in unique_item_cats]
        ax_item_cov.legend(handles=item_legend_elements, title='Item Covariates',
                            bbox_to_anchor=(0.25, -5), loc='center right')    

    plt.show()