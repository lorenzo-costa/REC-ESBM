from baseline import Baseline
import numpy as np
from math import lgamma

from nb_functions import sampling_scheme, compute_log_prob, compute_log_probs_cov
#########################
# ESBM 
#######################
# Inputs:
#     - number of users (required)
#     - number of items (required)
#     - n_clusters_users: number of clusters in the users (if None default to num_users)
#     - n_clusters_items: number of clusters in the items (if None default to num_items)
#     - prior_a: shape parameter for gamma prior (default to 1)
#     - prior_b: rate parameter for gamma prior (defaults to 1)
#     - user_clustering: cluster structure for users. if 'random' generate it from the prior
#       if None
#     - item_clustering: cluster structure for items (if None automatically assigned)
#     - Y: adjacency matrix (if None automatically generated)
#     - theta: mean parameter for the Poisson distribution (if None automatically generated)
#     - scheme_type: prior type. Possible choices are DP (dirichlet process), PY (pitman-yor process), GN (gnedin process), DM (dirichlet-multinomial model)
#     - scheme_param: additional parameter for cluster prior
#     - sigma: sigma parameter for Gibbs-type prior
#     - bar_h_users: maximum number of clusters for DM model
#     - bar_h_items: maximum number of clusters for DM model
#     - gamma: degree-correction factor (relevant only for DC model)
#     - cov_users: covariates for users. format should be a list of tuples (covname_covtype, covvalues) 
#       (note only cov type supported is categorical)
#     - cov_items: covariates for item. format should be a list of tuples (covname_covtype, covvalues)
#       (note only cov type supported is categorical)
#     - alpha: vector of additional parameters for covariate model (if int defaults to vector of equal numbers) 
 
class esbm(Baseline):
    def __init__(self, num_items, num_users, n_clusters_items=None, n_clusters_users=None,
                 prior_a=1, prior_b=1, seed = 42, user_clustering=None, item_clustering=None,
                 Y=None, theta = None, scheme_type = None, scheme_param = None,
                 sigma = None, bar_h_users=None, bar_h_items=None, gamma=None,
                 epsilon = 1e-6, verbose_users=False, verbose_items = False, device='cpu', 
                 cov_users=None, cov_items=None, alpha_c=1, degree_param_users=1, degree_param_items=1):

        super().__init__(num_items=num_items, num_users=num_users, n_clusters_items=n_clusters_items, n_clusters_users=n_clusters_users,
                 prior_a=prior_a, prior_b=prior_b, seed = seed, user_clustering=user_clustering, item_clustering=item_clustering,
                 Y=Y, theta = theta, scheme_type = scheme_type, scheme_param = scheme_param, sigma = sigma, bar_h_users=bar_h_users,
                 bar_h_items=bar_h_items, gamma=gamma, epsilon = epsilon, verbose_items=verbose_items, verbose_users=verbose_users,
                 device=device, alpha_c=alpha_c, cov_users=cov_users, cov_items=cov_items, 
                 degree_param_users=degree_param_users, degree_param_items=degree_param_items)
    
    #########
    # main computation, performs a single gibbs step for both users and items  
    def gibbs_step(self):
        frequencies_users = self.frequencies_users
        frequencies_items = self.frequencies_items         
        
        #################################################
        # step for users
        #################################################

        mhk = self.compute_mhk()
        yuk = self.compute_yuk()

        H = self.n_clusters_users
        V = self.num_users        
        nch = self.cov_nch_users
        
        for u in range(self.num_users):
            if self.verbose_users is True:
                print('\n', self.user_clustering, u)
                print('H ', H)
                print('frequencies', frequencies_users)
                print('nch', nch)
            
            cluster_user = self.user_clustering[u]
            frequencies_users_minus = frequencies_users.copy()
            
            # "remove" user from cluster
            frequencies_users_minus[cluster_user] -= 1
            frequencies_users[cluster_user] -= 1
            
            if nch is not None:
                nch_minus = []
                for cov in range(len(nch)):
                    c = self.cov_values_users[cov][u]
                    nch_minus.append(nch[cov].copy())
                    nch_minus[-1][c, cluster_user] -= 1
            
            # if the current cluster becomes empty, remove that row
            if frequencies_users_minus[cluster_user] == 0:
                mhk_minus = np.vstack([mhk[:cluster_user], mhk[cluster_user+1:]])
                frequencies_users_minus = np.concatenate([frequencies_users_minus[:cluster_user], frequencies_users_minus[cluster_user+1:]])
                H -= 1
                if nch is not None:
                    for cov in range(len(nch)):
                        nch_minus[cov] = np.hstack([nch[cov][:, :cluster_user], nch[cov][:, cluster_user+1:]])
            # else copy mhk and comoute mhk without u
            else:
                mhk_minus = mhk.copy()
                mhk_minus[cluster_user] -= yuk[u]

            # compute prior part
            probs = sampling_scheme(V, H, frequencies=frequencies_users_minus, bar_h=self.bar_h_users, scheme_type=self.scheme_type, 
                                    scheme_param=self.scheme_param, sigma=self.sigma, gamma=self.gamma)
            
            #compute main adj matrix part
            log_probs = compute_log_prob(probs, mhk_minus=mhk_minus, frequencies_primary_minus=frequencies_users_minus, frequencies_secondary=frequencies_items,
                                     y_values=np.ascontiguousarray(yuk[u]), epsilon=self.epsilon, a=self.prior_a, b=self.prior_b, max_clusters=H, 
                                     is_user_mode=True, degree_corrected=False, degree_param=1, degree_cluster_minus=[1], degree_node=1, device=self.device)
            
            # compute covs part       
            log_probs_cov = 0
            if nch is not None:
                log_probs_cov = compute_log_probs_cov(probs, idx=u, cov_types=self.cov_types_users, cov_nch = nch_minus, cov_values = self.cov_values_users, 
                                                nh=frequencies_users_minus, alpha_c = self.alpha_c, alpha_0 = self.alpha_0)
            
            # sum and use exp trick for stability
            probs = np.log(probs+self.epsilon)+log_probs + log_probs_cov
            probs = np.exp(probs-max(probs))
            probs /= probs.sum()
            
            # choose cluster assignment
            assignment = np.random.choice(len(probs), p=probs)
            
            if self.verbose_users is True:
                print('assignment', assignment)
            # if assignment is the same restore quantities
            if assignment == cluster_user:
                if frequencies_users[cluster_user] == 0:
                    H += 1
                frequencies_users[cluster_user] += 1
                if self.verbose_users is True:
                    print('same cluster, do nothing')
            
            # changing cluster
            else:
                if frequencies_users[cluster_user]==0:
                    self.user_clustering[np.where(self.user_clustering>= cluster_user)] -= 1
                    
                # creating new (prev empty) cluster
                if assignment >= H:
                    if self.verbose_users is True:
                        print('assigning to new cluster: ', assignment)
                        print('old cluster: ', cluster_user)
                    self.user_clustering[u] = assignment

                    #updating quantities
                    H += 1
                    frequencies_users_minus = np.append(frequencies_users_minus, 1)

                    # updating mhk (yuk not changed since item clusters unchanged)
                    mhk = np.vstack([mhk_minus, yuk[u]])
                    
                    #update nch
                    if nch is not None:
                        for cov in range(len(nch)):
                            c = self.cov_values_users[cov][u]
                            nch_minus[cov] = np.column_stack([nch_minus[cov], np.zeros(nch_minus[cov].shape[0])])
                            nch_minus[cov][c, assignment] += 1
                            
                        nch = nch_minus
                
                # into old cluster
                else:
                    if self.verbose_users is True:
                        print('adding to old: ', assignment)
                    frequencies_users_minus[assignment] += 1
                    self.user_clustering[u] = assignment
                    mhk_minus[assignment] += yuk[u]
                    mhk = mhk_minus
                    
                    if nch is not None:
                        for cov in range(len(nch)):
                            c = self.cov_values_users[cov][u]
                            nch_minus[cov][c, assignment] += 1
                        nch = nch_minus
                frequencies_users = frequencies_users_minus
        
        self.cov_nch_users = nch
        self.n_clusters_users = H
        self.frequencies_users = frequencies_users

        ################################################
        # step for items
        ################################################
        K = self.n_clusters_items
        V = self.num_items

        yih = self.compute_yih()
        mhk = self.compute_mhk()
        nch = self.cov_nch_items

        for i in range(self.num_items):
            if self.verbose_items is True:
                print('\n', self.item_clustering, i)
                print('frequencies', frequencies_items)
                print('K ', K)
            
            cluster_item = self.item_clustering[i]
            frequencies_items_minus = frequencies_items.copy()

            # remove item from cluster
            frequencies_items_minus[cluster_item] -= 1
            frequencies_items[cluster_item] -= 1
            
            if nch is not None:
                nch_minus = []
                for cov in range(len(nch)):
                    c = self.cov_values_items[cov][i]
                    nch_minus.append(nch[cov].copy())
                    nch_minus[-1][c, cluster_item] -= 1
                    
            # if cluster becomes empty remove that row
            if frequencies_items_minus[cluster_item]==0:
                mhk_minus = np.hstack([mhk[:, :cluster_item],mhk[:, cluster_item+1:]])
                frequencies_items_minus = np.concatenate([frequencies_items_minus[:cluster_item], frequencies_items_minus[cluster_item+1:]])
                K -= 1
                if nch is not None:
                    for cov in range(len(nch)):
                        nch_minus[cov] = np.hstack([nch[cov][:, :cluster_item], nch[cov][:, cluster_item+1:]])
            # else simply compute mhk minus
            else:
                mhk_minus = mhk.copy()
                mhk_minus[:, cluster_item] -= yih[i]

            # contribution of prior
            probs = sampling_scheme(V, K, frequencies=frequencies_items_minus, bar_h=self.bar_h_items, scheme_type=self.scheme_type, scheme_param=self.scheme_param, sigma=self.sigma, gamma=self.gamma)
            
            # contribution of adj matrix
            log_probs = compute_log_prob(probs, mhk_minus=mhk_minus, frequencies_primary_minus=frequencies_items_minus, frequencies_secondary=frequencies_users,
                                     y_values=np.ascontiguousarray(yih[i]), epsilon=self.epsilon, a=self.prior_a, b=self.prior_b, max_clusters=K, 
                                     is_user_mode=False, degree_corrected=False, degree_param=1, degree_cluster_minus=[1], degree_node=1, device=self.device)
            
            # contribution of covs
            log_probs_cov = 0
            if nch is not None:
                log_probs_cov = compute_log_probs_cov(probs, idx=i, cov_types=self.cov_types_items, cov_nch = nch_minus, cov_values = self.cov_values_items, 
                                                nh=frequencies_items_minus, alpha_c = self.alpha_c, alpha_0 = self.alpha_0)
            
            # sum and use exp trick for stability               
            probs = np.log(probs+self.epsilon)+log_probs+log_probs_cov
            probs = np.exp(probs-max(probs))
            probs /= probs.sum()

            # choose cluster assignment
            assignment = np.random.choice(len(probs), p=probs)
            
            # if assignment is the same restore quantities
            if assignment == cluster_item:
                if frequencies_items[cluster_item]==0:
                    K += 1
                frequencies_items[cluster_item] += 1
                if self.verbose_items is True:
                    print('adding to the same, do nothing')
                    
            # changing cluster
            else:
                if frequencies_items[cluster_item]==0:
                    self.item_clustering[np.where(self.item_clustering>= cluster_item)] -= 1
                
                # creating a new (prev empty) cluster
                if assignment >= K:
                    if self.verbose_items is True:
                        print("assigning to new cluster: ", assignment)
                        print('old cluster: ', cluster_item)

                    self.item_clustering[i] = assignment
                    K += 1
                    frequencies_items_minus = np.append(frequencies_items_minus, 1)

                    # updating mhk (yuk not changed since item clusters unchanged)
                    mhk = np.column_stack([mhk_minus, yih[i]])
                    
                    #update nch
                    if nch is not None:
                        for cov in range(len(nch)):
                            c = self.cov_values_items[cov][i]
                            nch_minus[cov] = np.column_stack([nch_minus[cov], np.zeros(nch_minus[cov].shape[0])])
                            nch_minus[cov][c, assignment] += 1
                        nch = nch_minus
                        
                # putting into a prev existing cluster
                else:
                    if self.verbose_items is True:
                        print('adding to old: ', assignment)

                    frequencies_items_minus[assignment] += 1
                    self.item_clustering[i] = assignment

                    # previous connections already removed, add the new ones
                    mhk_minus[:, assignment] += yih[i]
                    mhk = mhk_minus
                    if nch is not None:
                        for cov in range(len(nch)):
                            c = self.cov_values_items[cov][i]
                            nch_minus[cov][c, assignment] += 1
                        nch = nch_minus
                        
                frequencies_items = frequencies_items_minus
                
        self.cov_nch_items = nch
        self.n_clusters_items = K
        self.frequencies_items = frequencies_items
        return
    
    ############
    # estimating theta from posterior comutations
    def estimate_theta(self):
        if self.estimated_users is None or self.estimated_items is None:
            raise Exception('cluster assignment must be estimated first')
        
        mhk = self.compute_mhk(self.user_clustering, self.item_clustering)
        theta = (self.prior_a + mhk) / (self.prior_b + np.outer(self.frequencies_users, self.frequencies_items))
        
        self.estimated_theta = theta
        return theta
    
    #########
    # generate point prediction (rating value) for pairs of user-item
    # (uses the theta value for user-item cluster)
    def point_predict(self, pairs, seed=None):
        if seed is None:
            np.random.seed(self.seed)
        elif seed == -1:
            pass
        else:
            np.random.seed(seed)
        
        if self.estimated_users is None or self.estimated_items is None:
            raise Exception('cluster assignment must be estimated first')
        if self.estimated_theta is None:
            self.estimate_theta()
        
        if not isinstance(pairs, list):
            pairs = [pairs]
        
        preds = []
        for u, i in pairs:
            zu = self.user_clustering[u]
            qi = self.item_clustering[i]
            
            # predict with predictive posterior mean
            preds.append(self.estimated_theta[zu, qi])

        return preds
    
    #############
    # for a list of users returns items in cluster with highest score 
    # (highest theta value)
    def predict_with_ranking(self, users):
        if self.estimated_users is None or self.estimated_items is None:
            raise Exception('cluster assignment must be estimated first')
        
        if self.estimated_theta is None:
            self.estimate_theta()
        
        theta = self.estimated_theta
        
        out = []
        for u in users:
            zu = self.user_clustering[u]
            preds = theta[zu]
            top_cluster = np.argsort(preds)[::-1][0]
            top_items = np.arange(self.num_items)[np.where(self.item_clustering==top_cluster)]
            unseen_items = np.where(self.Y[u] == 0)[0]
            unseen_items_intersect = np.intersect1d(top_items, unseen_items)
            out.append(unseen_items_intersect)
        
        self.predict_mode = 'ranking'
        return out
    
    ###########
    # return k recommended items for a list of users.
    # mode parameter scpecifies how these are selected
    # - 'random': random choice from cluster with highest score
    # - 'popularity': selects the most popular items (i.e. most reviewed) 
    #    in the cluster with highest score
    def predict_k(self, users, k=10, mode='random', seed=42):
        np.random.seed(seed)
        if self.estimated_users is None or self.estimated_items is None:
            raise Exception('cluster assignment must be estimated first')
        
        if self.estimated_theta is None:
            self.estimate_theta()
        
        if mode == 'popularity':
            degree_items = self.Y.sum(axis=0)
            
        theta = self.estimated_theta
        out = []
        for u in users:
            zu = self.user_clustering[u]
            preds = theta[zu]
            top_cluster = np.argsort(preds)[::-1][0]
            top_items = np.arange(self.num_items)[np.where(self.item_clustering==top_cluster)]
            unseen_items = np.where(self.Y[u] == 0)[0]
            unseen_items_intersect = np.intersect1d(top_items, unseen_items)
            i = 1
            while len(unseen_items_intersect) < k:
                top_cluster = np.argsort(preds)[::-1][i]
                top_items = np.arange(self.num_items)[np.where(self.item_clustering==top_cluster)]
                unseen_items_intersect_ext = np.intersect1d(top_items, unseen_items)
                to_pick = k-len(unseen_items_intersect)
                if len(unseen_items_intersect_ext) < to_pick:
                    unseen_items_intersect_add = unseen_items_intersect_ext
                else:
                    unseen_items_intersect_add = np.random.choice(unseen_items_intersect_ext, k-len(unseen_items_intersect), replace=False)
                unseen_items_intersect = np.concatenate([unseen_items_intersect, unseen_items_intersect_add])
                i += 1
                
            if len(unseen_items_intersect) < k:
                k = len(unseen_items_intersect)
                print('not enough candidates, reducing k to ', k)
            
            if mode == 'random':
                out.append(np.random.choice(unseen_items_intersect, k, replace=False))
            if mode == 'popularity':
                degree_candindates = degree_items[unseen_items_intersect]
                top_items = np.argsort(degree_candindates)[::-1][0:k]
                out.append(unseen_items_intersect[top_items])
                
        self.predict_mode = 'k'
        return out
    
    ############
    # compute log-likelihood for each edge at a given iteration
    def compute_llk(self, iter):
        np.random.seed(self.seed)
        
        clustering_users = self.mcmc_draws_users[iter]
        clustering_items = self.mcmc_draws_items[iter]
        frequencies_users = self.mcmc_draws_users_frequencies[iter]
        frequencies_items = self.mcmc_draws_items_frequencies[iter]
        
        mhk = self.compute_mhk(clustering_users, clustering_items)
        # first sample theta
        theta = np.random.gamma(self.prior_a+mhk, 
                                1/(self.prior_b+np.outer(frequencies_users, frequencies_items)))
        
        # log likelihood
        llk_out = []
        for u in range(self.num_users):
            for i in range(self.num_items):
                zu = clustering_users[u]
                qi = clustering_items[i]
                llk_out.append(self.Y[u,i]*np.log(theta[zu, qi])-theta[zu,qi]-lgamma(self.Y[u,i]+1))
        return llk_out