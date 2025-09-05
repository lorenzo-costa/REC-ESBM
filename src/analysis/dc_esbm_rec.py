from esbm_rec import esbm
import numpy as np
from analysis.numba_functions import sampling_scheme, compute_log_prob, compute_log_probs_cov, compute_log_likelihood
import time
from math import lgamma

class dcesbm(esbm):
    """Degree-Corrected Exteneded Stochastic Block Model
    
    Degree corrected version of the bipartite Extended Stochastic Block Model (ESBM) 
    with Poisson likelihood.

    Parameters
    ----------
    num_items : int
        number of items
    num_users : int
        number of users
    user_clustering : list or array-like
        cluster assignments for users, by default None. If 'random' generate it from the prior, if None
        assign each user to its own cluster
    item_clustering : list or array-like
        cluster assignments for items, by default None. If 'random' generate it from the prior, if None
        assign each item to its own cluster
    Y : 2D array
        adjacency matrix (if None automatically generated), by default None
    theta : (2D array)
        mean parameter for the Poisson distribution (if None automatically generated)
    prior_a : float
        shape parameter for gamma prior, by default 1
    prior_b : float
        rate parameter for gamma prior, by default 1
    scheme_type : str
        prior type. Possible choices are DP (dirichlet process), PY (pitman-yor process), 
        GN (gnedin process), DM (dirichlet-multinomial model)
    scheme_param : float
        additional parameter for cluster prior, by default None
    sigma : float
        sigma parameter for Gibbs-type prior, by default None
    gamma : float
        additional parameter for GN model, by default None
    bar_h_users : int
        maximum number of clusters for DM model, by default None
    bar_h_items : int
        maximum number of clusters for DM model, by default None
    degree_param_users : float
        degree-correction parameter for users (relevant only for DC model), by default 1
    degree_param_items : float
        degree-correction parameter for items (relevant only for DC model), by default 1
    alpha_c : float or list
        additional parameter for categorical covariate model (if int defaults to vector of equal numbers), by default 1
    cov_users : list
        list of tuples (covname_covtype, covvalues) for user covariates, by default None
    cov_items : list
        list of tuples (covname_covtype, covvalues) for item covariates, by default None
    device : str
        device to use (cpu or gpu), by default 'cpu'
    
    Attributes
    ----------
    Y : 2D array
        adjacency matrix
    num_items : int
        number of items
    num_users : int
        number of users
    n_clusters_users : int
        number of clusters in the users
    n_clusters_items : int
        number of clusters in the items
    train_llk : 1D array 
        log-likelihood values during training
    mcmc_draws_users : 2D array
        MCMC samples of user cluster assignments during training
    mcmc_draws_items : 2D array
        MCMC samples of item cluster assignments during training
    estimated_items : str
        method used for estimating item clusters
    estimated_users : str
        method used for estimating user clusters
    estimated_theta : 2D array
        estimated mean parameter for the Poisson distribution
    """
    def __init__(self,
                 *, 
                 num_items=None, 
                 num_users=None, 
                 user_clustering=None, 
                 item_clustering=None,
                 Y=None, 
                 theta = None,
                 prior_a=1, 
                 prior_b=1, 
                 scheme_type = None, 
                 scheme_param = None, 
                 sigma = None, 
                 gamma=None,
                 bar_h_users=None, 
                 bar_h_items=None, 
                 degree_param_users=1, 
                 degree_param_items=1,
                 alpha_c=1, 
                 cov_users=None, 
                 cov_items=None,
                 epsilon = 1e-6,
                 seed = 42,  
                 verbose_users=False, 
                 verbose_items = False, 
                 device='cpu'):
        
        # initialise phi to None
        self.estimated_phi_users = None
        self.estimated_phi_items = None
        
        kwargs = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
        super().__init__(**kwargs)
 
        # compute degree of each user and each cliuster from cluster structure
        self.degree_users = self.Y.sum(axis=1)
        self.degree_items = self.Y.sum(axis=0)
        
        self.degree_cluster_users = self.compute_degree(clustering=self.user_clustering, 
                                                        degrees=self.degree_users, 
                                                        n_clusters=self.n_clusters_users)
        
        self.degree_cluster_items = self.compute_degree(clustering=self.item_clustering, 
                                                        degrees=self.degree_items, 
                                                        n_clusters=self.n_clusters_items)
    
    ########
    # function to compute cluster degree
    def compute_degree(self, clustering, degrees, n_clusters):
        degree = np.zeros(n_clusters)
        for i in range(len(clustering)):
            degree[clustering[i]] += degrees[i]
        return degree
    
    ############
    # modified version of generate data to take degree correction into account
    def generate_data(self):
        np.random.seed(self.seed)
        
        phi_u = np.zeros(shape=(self.n_clusters_users, self.num_users))
        for h in range(self.n_clusters_users):
            idx = np.where(self.user_clustering == h)[0]
            param = np.ones(len(idx)) * self.degree_param_users
            phi_u[h, idx] = np.random.dirichlet(param)
        
        phi_i = np.zeros(shape=(self.n_clusters_items, self.num_items))
        for k in range(self.n_clusters_items):
            idx = np.where(self.item_clustering == k)[0]
            param = np.ones(len(idx)) * self.degree_param_items
            phi_i[k, idx] = np.random.dirichlet(param)
        
        self.phi_u = phi_u.copy()
        self.phi_i = phi_i.copy()

        Y_params = np.zeros(shape=(self.num_users, self.num_items))
        for i in range(self.num_users):
            for j in range(self.num_items):
                zu = self.user_clustering[i]
                qi = self.item_clustering[j]
                eta_u = self.frequencies_users[zu] * self.phi_u[zu, i]
                eta_i = self.frequencies_items[qi] * self.phi_i[qi, j]
                Y_params[i, j] = self.theta[zu, qi] * eta_u * eta_i

        Y = np.random.poisson(Y_params)
        self.Y = Y.copy()
        return
        
    def gibbs_step(self):
        frequencies_users = self.frequencies_users
        frequencies_items = self.frequencies_items      
        degree_cluster_users = self.degree_cluster_users 
        degree_cluster_items = self.degree_cluster_items 
        degree_param_users = self.degree_param_users
        degree_param_items = self.degree_param_items
        
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
                print('cluster degree', degree_cluster_users)
            
            cluster_user = self.user_clustering[u]
            frequencies_users_minus = frequencies_users.copy()

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
            else:
                mhk_minus = mhk.copy()
                mhk_minus[cluster_user] -= yuk[u]
            
            degree_cluster_user_minus = mhk_minus.sum(axis = 1)
            
            probs = sampling_scheme(V, H, frequencies=frequencies_users_minus, bar_h=self.bar_h_users, scheme_type=self.scheme_type, 
                                    scheme_param=self.scheme_param, sigma=self.sigma, gamma=self.gamma)

            log_probs = compute_log_prob(probs, 
                                         mhk_minus = mhk_minus, 
                                         frequencies_primary_minus=frequencies_users_minus, 
                                         frequencies_secondary=frequencies_items, 
                                         y_values = np.ascontiguousarray(yuk[u]), 
                                         max_clusters=H, 
                                         epsilon=self.epsilon, 
                                         a=self.prior_a, 
                                         b=self.prior_b,  
                                         device=self.device, 
                                         degree_corrected=True,
                                         degree_cluster_minus=degree_cluster_user_minus, 
                                         degree_node=self.degree_users[u], 
                                         degree_param=degree_param_users,
                                         is_user_mode=True) 
            
            log_probs_cov = 0
            if nch is not None:
                log_probs_cov = compute_log_probs_cov(probs, 
                                                      idx=u, 
                                                      cov_types=self.cov_types_users, 
                                                      cov_nch=nch_minus, 
                                                      cov_values=self.cov_values_users, 
                                                      nh=frequencies_users_minus, 
                                                      alpha_c=self.alpha_c, 
                                                      alpha_0=self.alpha_0)

            probs = np.log(probs+self.epsilon)+log_probs + log_probs_cov
            probs = np.exp(probs-max(probs))
            probs /= probs.sum()
            
            # choose cluster assignment
            assignment = np.random.choice(len(probs), p=probs)
            
            if self.verbose_users is True:
                print('assignment', assignment)
            # if assignment is the same do nothing
            if assignment == cluster_user:
                if frequencies_users[cluster_user] == 0:
                    H += 1
                frequencies_users[cluster_user] += 1
                if self.verbose_users is True:
                    print('same cluster, do nothing')
            else:
                # new cluster
                if frequencies_users[cluster_user]==0:
                    self.user_clustering[np.where(self.user_clustering>= cluster_user)] -= 1

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
                degree_cluster_users = mhk.sum(axis=1)
        
        self.degree_cluster_users = degree_cluster_users
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
                print('degree cluster', degree_cluster_items)
            
            cluster_item = self.item_clustering[i]
            frequencies_items_minus = frequencies_items.copy()

            frequencies_items_minus[cluster_item] -= 1
            frequencies_items[cluster_item] -= 1
            
            if nch is not None:
                nch_minus = []
                for cov in range(len(nch)):
                    c = self.cov_values_items[cov][i]
                    nch_minus.append(nch[cov].copy())
                    nch_minus[-1][c, cluster_item] -= 1

            if frequencies_items_minus[cluster_item]==0:
                mhk_minus = np.hstack([mhk[:, :cluster_item],mhk[:, cluster_item+1:]])
                frequencies_items_minus = np.concatenate([frequencies_items_minus[:cluster_item], frequencies_items_minus[cluster_item+1:]])
                K -= 1
                if nch is not None:
                    for cov in range(len(nch)):
                        nch_minus[cov] = np.hstack([nch[cov][:, :cluster_item], nch[cov][:, cluster_item+1:]])
            else:
                mhk_minus = mhk.copy()
                mhk_minus[:, cluster_item] -= yih[i]
            
            degree_cluster_item_minus = mhk_minus.sum(axis=0)

            probs = sampling_scheme(V, K, frequencies=frequencies_items_minus, bar_h=self.bar_h_items, scheme_type=self.scheme_type, scheme_param=self.scheme_param, sigma=self.sigma, gamma=self.gamma)

            log_probs = compute_log_prob(probs=probs, mhk_minus=mhk_minus, frequencies_primary_minus=frequencies_items_minus, 
                                     frequencies_secondary = frequencies_users, y_values=np.ascontiguousarray(yih[i]), max_clusters=K,
                                     epsilon=self.epsilon, a = self.prior_a, b=self.prior_b,  device=self.device, degree_corrected=True,
                                     degree_cluster_minus=degree_cluster_item_minus, degree_node=self.degree_items[i],
                                     degree_param=degree_param_items, is_user_mode=False)
            
            log_probs_cov = 0
            if nch is not None:
                log_probs_cov = compute_log_probs_cov(probs, idx=i, cov_types=self.cov_types_items, cov_nch = nch_minus, cov_values = self.cov_values_items, 
                                                nh=frequencies_items_minus, alpha_c = self.alpha_c, alpha_0 = self.alpha_0)
                                
            probs = np.log(probs+self.epsilon)+log_probs+log_probs_cov
            probs = np.exp(probs-max(probs))
            probs /= probs.sum()

            # choose cluster assignment
            assignment = np.random.choice(len(probs), p=probs)
            if self.verbose_items is True:
                print('assignment', assignment)
            
            # if assignment is the same do nothing
            if assignment == cluster_item:
                if frequencies_items[cluster_item]==0:
                    K += 1
                frequencies_items[cluster_item] += 1
                if self.verbose_items is True:
                    print('adding to the same, do nothing')
            else:
                if frequencies_items[cluster_item]==0:
                    self.item_clustering[np.where(self.item_clustering>= cluster_item)] -= 1

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
                else:
                    if self.verbose_items is True:
                        print('adding to old: ', assignment)
                    frequencies_items_minus[assignment] += 1
                    self.item_clustering[i] = assignment
                    mhk_minus[:, assignment] += yih[i]
                    mhk = mhk_minus
                    
                    if nch is not None:
                        for cov in range(len(nch)):
                            c = self.cov_values_items[cov][i]
                            nch_minus[cov][c, assignment] += 1
                        nch = nch_minus
                        
                frequencies_items = frequencies_items_minus
                degree_cluster_items = mhk.sum(axis=0)
        
        self.degree_cluster_items = degree_cluster_items
        self.cov_nch_items = nch
        self.n_clusters_items = K
        self.frequencies_items = frequencies_items
        return
    
    def gibbs_train(self, n_iters, verbose=0, warm_start=False, degree_corrected=True):
        np.random.seed(self.seed)
        
        self.n_iters = n_iters
        assert len(self.user_clustering)==len(self.degree_users)
        
        ll = compute_log_likelihood(nh = self.frequencies_users, 
                                    nk = self.frequencies_items, 
                                    a = self.prior_a, 
                                    b = self.prior_b, 
                                    eps = self.epsilon, 
                                    mhk=self.compute_mhk(), 
                                    user_clustering=self.user_clustering, 
                                    item_clustering=self.item_clustering,
                                    degree_param_users=self.degree_param_users,
                                    degree_param_items=self.degree_param_items,
                                    dg_u=self.degree_users, 
                                    dg_i=self.degree_items, 
                                    dg_cl_i=self.degree_cluster_items, 
                                    dg_cl_u=self.degree_cluster_users, 
                                    degree_corrected=degree_corrected)
        
        print('starting log likelihood', ll)
        
        if warm_start is True:
            raise Exception('implement this')
        else:
            llks = np.zeros(n_iters+1)
            user_cluster_list = np.zeros((n_iters+1, self.num_users), dtype=np.int32)
            item_cluster_list = np.zeros((n_iters+1, self.num_items), dtype=np.int32)
            frequencies_users_list = []
            frequencies_items_list = []
            degree_users_list = []
            degree_items_list = []
            
            llks[0] = ll
            user_cluster_list[0] = self.user_clustering.copy()
            item_cluster_list[0] = self.item_clustering.copy()
            frequencies_users_list.append(self.frequencies_users.copy())
            frequencies_items_list.append(self.frequencies_items.copy())
            degree_users_list.append(self.degree_users)
            degree_items_list.append(self.degree_items)
        
        check = time.time()
        for it in range(n_iters):
            
            self.gibbs_step()
            ll = compute_log_likelihood(nh = self.frequencies_users, 
                                        nk = self.frequencies_items,
                                        a = self.prior_a, 
                                        b = self.prior_b, 
                                        eps = self.epsilon,
                                        mhk = self.compute_mhk(),
                                        user_clustering=self.user_clustering, 
                                        item_clustering=self.item_clustering,
                                        degree_param_users=self.degree_param_users,
                                        degree_param_items=self.degree_param_items,
                                        dg_u=self.degree_users, 
                                        dg_i=self.degree_items, 
                                        dg_cl_i=self.degree_cluster_items, 
                                        dg_cl_u=self.degree_cluster_users, 
                                        degree_corrected=degree_corrected)
            
            llks[it+1] += ll
            user_cluster_list[it+1] += self.user_clustering
            item_cluster_list[it+1] += self.item_clustering
            frequencies_users_list.append(self.frequencies_users.copy())
            frequencies_items_list.append(self.frequencies_items.copy())
            degree_users_list.append(self.degree_users.copy())
            degree_items_list.append(self.degree_items.copy())
            
            
            if verbose >= 1:
                if it % (n_iters // 10) == 0:
                    print(it, llks[it+1])
                    print('time', time.time()-check)
                    check = time.time()
                if verbose >= 2:
                    if it % (n_iters // 10) == 0:
                        print(it)
                        print('user freq ', self.frequencies_users)
                        print('ite freq ', self.frequencies_items)
                    if verbose >= 3:
                        print('user cluser ', self.user_clustering)
                        print('item cluster ', self.item_clustering)
        
        print('end llk: ', llks[-1])
        self.train_llk = llks
        self.mcmc_draws_users = user_cluster_list
        self.mcmc_draws_items = item_cluster_list
        self.mcmc_draws_users_frequencies = frequencies_users_list
        self.mcmc_draws_items_frequencies = frequencies_items_list
        self.mcmc_draws_degree_users = degree_users_list
        self.mcmc_draws_degree_items = degree_items_list
        
        return llks, user_cluster_list, item_cluster_list
    
    def estimate_phi(self):
        phi_users = np.zeros(shape=(self.n_clusters_users, self.num_users))
        for h in range(self.n_clusters_users):
            idx = np.where(self.user_clustering==h)
            temp = self.degree_users[idx] + self.degree_param_users
            phi_users[h, idx] = temp/temp.sum()
        
        phi_items = np.zeros(shape=(self.n_clusters_items, self.num_items))
        for k in range(self.n_clusters_items):
            idx = np.where(self.item_clustering==k)
            temp = self.degree_items[idx] + self.degree_param_items
            phi_items[k, idx] = temp/temp.sum()
            
        self.estimated_phi_users = phi_users
        self.estimated_phi_items = phi_items
    
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
        if self.estimated_phi_users is None or self.estimated_phi_items is None:
            self.estimate_phi()
            
        if not isinstance(pairs, list):
            pairs = [pairs]
        
        preds = []
        for u, i in pairs:
            zu = self.user_clustering[u]
            qi = self.item_clustering[i]
            
            eta_u = self.frequencies_users[zu]*self.estimated_phi_users[zu, u]
            eta_i = self.frequencies_items[qi]*self.estimated_phi_items[qi, i]
            # predict with predictive posterior mean
            preds.append(self.estimated_theta[zu, qi]*eta_u*eta_i)

        return preds
        
    def predict_k(self, users, k=10, seed=42, ignore_seen=True):
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
        if self.estimated_phi_users is None or self.estimated_phi_items is None:
            self.estimate_phi()
            
        theta = self.estimated_theta
        out = []
        
        for u in users:
            if ignore_seen is True:
                unseen_items = np.where(self.Y[u] == 0)[0]
            else:
                unseen_items = np.arange(self.Y.shape[0])
                
            scores = np.zeros(self.num_items)
            zu = self.user_clustering[u]
            for i in unseen_items:
                qi = self.item_clustering[i]
                scores[i] = self.frequencies_items[qi]*self.estimated_phi_items[qi, i]*theta[zu, qi]
            
            top_items = np.argsort(scores)[::-1][0:k]
            out.append(top_items)
                
        self.predict_mode = 'k'
        return out