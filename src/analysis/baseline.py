import numpy as np
import math
from scipy import sparse
from scipy.stats import mode

from nb_functions import sampling_scheme, compute_log_probs_cov, compute_log_likelihood
from vi_functs import minVI
from nb_functions import compute_co_clustering_matrix


#########################################
# baseline class
########################################

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
    
class Baseline:
    def __init__(self, num_items, num_users, 
                 n_clusters_items=None, n_clusters_users=None,
                 prior_a=1, prior_b=1, 
                 seed = 42, 
                 user_clustering=None, item_clustering=None,
                 Y=None, theta = None, 
                 alpha_c=1, cov_users=None, cov_items=None,
                 scheme_type = None, scheme_param = None, sigma = None, 
                 bar_h_users=None, bar_h_items=None, 
                 gamma=None,
                 epsilon = 1e-6, verbose_users=False, verbose_items = False, 
                 device='cpu', 
                 degree_param_users=1, degree_param_items=1):
        
        ###########################
        #argument checks
        if n_clusters_items is None:
            n_clusters_items = num_items
        if n_clusters_users is None:
            n_clusters_users = num_users

        if scheme_type is None:
            raise Exception('please provide scheme type')
        
        if scheme_type == 'DM':
            if not isinstance(bar_h_users, int) or (bar_h_users < 0) or (bar_h_users > num_users):
                raise Exception('provide valid maximum number of clusters users for DM)')
            if not isinstance(bar_h_items, int) or (bar_h_items < 0) or (bar_h_items>num_items):
                raise Exception('provide valid maximum number of clusters items for DM)')
            if not isinstance(sigma, (int, float)) or sigma >= 0:
                raise Exception('provide valid sigma (-item_clustering) parameter for DM')

        if scheme_type == 'DP':
            if not isinstance(scheme_param, (int, float)) or scheme_param <= 0:
                raise Exception('provide valid concentration parameter for DP')
        
        if scheme_type == 'PY':
            if not isinstance(sigma, (int, float)) or (sigma < 0 or sigma >= 1):
                raise Exception('provide valid sigma in [0, 1) for PY')
            if not isinstance(scheme_param, (int, float)) or scheme_param <= -sigma:
                raise Exception('provide valid user_clustering parameter for PY')
            if sigma == 0:
                print('note: for sigma=0 the PY reduces to DP, use scheme_type=DP for greater efficiency')
        if scheme_type == 'GN':
            if not isinstance(gamma, float) or (gamma<=0 or gamma>= 1):
                raise Exception('please provide valid gamma paramter for GN')

        self.seed = seed
        self.num_items = num_items
        self.num_users = num_users
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.verbose_items = verbose_items
        self.verbose_users = verbose_users

        self.epsilon = epsilon

        self.scheme_type = scheme_type
        self.scheme_param = scheme_param
        self.bar_h_users = bar_h_users
        self.bar_h_items = bar_h_items
        self.gamma = gamma
        self.sigma = sigma
        
        self.degre_param_users = degree_param_users
        self.degre_param_items = degree_param_items
        
        self.device = device
        self.alpha_c = alpha_c
        self.alpha_0 = np.sum(np.array(alpha_c))
        
        self.cov_users=cov_users
        self.cov_items=cov_items
        self.theta = theta
        
        self.train_llk = None
        self.mcmc_draws_users = None
        self.mcmc_draws_items = None
        
        self.estimated_items = None
        self.estimated_users = None
        
        self.estimated_theta = None
        
        self.llk_edges = None
        
        self.waic = None
        
        if cov_users is not None:
            self.cov_names_users, self.cov_types_users, self.cov_values_users = self.process_cov(cov_users)
    
        if cov_items is not None:
            self.cov_names_items, self.cov_types_items, self.cov_values_items = self.process_cov(cov_items)
        
        # if not clustering structure create it
        
        if user_clustering is None:
                self.user_clustering = [i for i in range(self.num_users)]    
        elif user_clustering == 'random':
            self.init_cluster_random('random', None)
        else:
            self.user_clustering=user_clustering
            
        if item_clustering is None:
            self.item_clustering = [i for i in range(self.num_items)]
        elif item_clustering=='random':
            self.init_cluster_random(None, 'random')
        else:
            self.item_clustering=item_clustering
        
        #computes cluster metrics needed later 
        self.process_clusters(self.user_clustering, self.item_clustering)
        
        # if not theta is provided generate it
        if self.theta is None:
            self.theta = np.random.gamma(1,1, size = (self.n_clusters_users, self.n_clusters_items))
        
        # if there are covs compute nch
        if cov_users is not None:
            self.cov_nch_users = self.compute_nch(self.cov_values_users, self.user_clustering, self.n_clusters_users)
        else:
            self.cov_nch_users = None
        if cov_items is not None:
            self.cov_nch_items = self.compute_nch(self.cov_values_items, self.item_clustering, self.n_clusters_items)
        else:
            self.cov_nch_items = None
            
        # if no adj matrix is given generate it
        if Y is None:
            print('randomly initialising data')
            self.generate_data()
        else:
            self.Y = Y
        
    ##########
    # function to compute cluster metrics
    def process_clusters(self, user_clustering, item_clustering):
        occupied_clusters_users, frequencies_users = np.unique(user_clustering, return_counts=True)
        self.frequencies_users = frequencies_users
        self.n_clusters_users = len(occupied_clusters_users)
        self.user_clustering = np.array(user_clustering)   
                    
        occupied_clusters_items, frequencies_items = np.unique(item_clustering, return_counts=True)
        self.frequencies_items = frequencies_items
        self.n_clusters_items = len(occupied_clusters_items)
        self.item_clustering = np.array(item_clustering)
    
    ########
    # function to generate adjacency matrix
    def generate_data(self):
        np.random.seed(self.seed)
        # Y params collects the appropriate theta entries accoridng to cluster structure
        Y_params = self.theta[self.user_clustering][:, self.item_clustering]
        # Y drawn from poisson distribution
        Y = np.random.poisson(Y_params)
        self.Y = Y.copy()
        return
    
    ##########
    # function to initialise clusters according to prior
    def init_cluster_random(self, user_clustering=None, item_clustering=None):
        np.random.seed(self.seed)
        if user_clustering == 'random':
            user_clustering = [0]
            H = 1
            V = 1
            users_frequencies = [1]
            print('initialsing user clusters random')
            nch_users = None
            if self.cov_users is not None:
                nch_users = []
                for cov in range(len(self.cov_values_users)):
                    n_unique = len(np.unique(self.cov_values_users[cov]))
                    temp = np.zeros(n_unique)
                    c = self.cov_values_users[cov][0]
                    temp[c] += 1
                    nch_users.append(temp.reshape(-1, 1))
                    
            for u in range(1, self.num_users):
                # order should be guaranteed by properties of dict in pyhton
                probs = sampling_scheme(V, H, users_frequencies, bar_h=self.bar_h_users, scheme_type=self.scheme_type,
                                        scheme_param=self.scheme_param, sigma=self.sigma, gamma=self.gamma)
                log_probs_cov = 0
                if nch_users is not None:
                    log_probs_cov = compute_log_probs_cov(probs, u, self.cov_types_users, nch_users, self.cov_values_users, 
                                                    users_frequencies, self.alpha_c, self.alpha_0)
                
                # convert back using exp and normalise
                log_probs = np.log(probs+self.epsilon)+log_probs_cov
                probs = np.exp(log_probs-max(log_probs))
                probs /= probs.sum()
                
                assignment = np.random.choice(len(probs), p=probs)
                if assignment >= H:
                    H += 1
                    users_frequencies.append(1)
                    
                    if nch_users is not None:
                        for cov in range(len(self.cov_values_users)):
                            n_unique = len(np.unique(self.cov_values_users[cov]))
                            temp = np.zeros(n_unique)
                            c = self.cov_values_users[cov][u]
                            temp[c] += 1
                            nch_users[cov] = np.column_stack([nch_users[cov], temp.reshape(-1, 1)])
                else:
                    users_frequencies[assignment] += 1
                    if nch_users is not None:
                        for cov in range(len(self.cov_values_users)):
                            c = self.cov_values_users[cov][u]
                            nch_users[cov][c, assignment] += 1
                            
                user_clustering.append(assignment)
                V += 1

            self.n_clusters_users = H
            self.frequencies_users = users_frequencies
            
            self.user_clustering = np.array(user_clustering)    
            self.n_clusters_users = len(np.unique(user_clustering))
        
        if item_clustering == 'random':
            item_clustering = [0]
            K = 1
            V = 1
            items_frequencies = [1]
            print('initialising item clusters random')
            nch_items = None
            if self.cov_items is not None:
                nch_items = []
                for cov in range(len(self.cov_values_items)):
                    n_unique = len(np.unique(self.cov_values_items[cov]))
                    temp = np.zeros(n_unique)
                    c = self.cov_values_items[cov][0]
                    temp[c] += 1
                    nch_items.append(temp.reshape(-1, 1))
                    
            for i in range(1, self.num_items):
                probs = sampling_scheme(V, K, items_frequencies, bar_h=self.bar_h_items, scheme_type=self.scheme_type,
                                        scheme_param=self.scheme_param, sigma=self.sigma, gamma=self.gamma)
                log_probs_cov = 0
                if nch_items is not None:
                    log_probs_cov = compute_log_probs_cov(probs, i, self.cov_types_items, nch_items, self.cov_values_items, 
                                                    items_frequencies, self.alpha_c, self.alpha_0)
                
                log_probs = np.log(probs+self.epsilon)+log_probs_cov
                probs = np.exp(log_probs-max(log_probs))
                probs /= probs.sum()
                            
                assignment = np.random.choice(len(probs), p=probs)
                if assignment >= K:
                    K += 1
                    items_frequencies.append(1)
                    
                    if nch_items is not None:
                        for cov in range(len(self.cov_values_items)):
                            n_unique = len(np.unique(self.cov_values_items[cov]))
                            temp = np.zeros(n_unique)
                            c = self.cov_values_items[cov][i]
                            temp[c] += 1
                            nch_items[cov] = np.column_stack([nch_items[cov], temp.reshape(-1, 1)])
                else:
                    items_frequencies[assignment] += 1
                    if nch_items is not None:
                        for cov in range(len(self.cov_values_items)):
                            c = self.cov_values_items[cov][i]
                            nch_items[cov][c, assignment] += 1
                            
                item_clustering.append(assignment)
                V += 1
                
                self.n_clusters_items = K
                self.frequencies_items = items_frequencies
            
            self.item_clustering = np.array(item_clustering)
            self.n_clusters_items = len(np.unique(item_clustering))
        
        return user_clustering, item_clustering
    
    ###########
    # compute mhk matrix suing fast sparse matrix multiplication
    def compute_mhk(self, user_clustering=None, item_clustering=None):
        if user_clustering is None:
            user_clustering = self.user_clustering
            num_users = self.num_users
            n_clusters_users = self.n_clusters_users
        else:
            num_users = len(self.user_clustering)
            n_clusters_users = len(np.unique(user_clustering))
            
        if item_clustering is None:
            item_clustering = self.item_clustering
            num_items = self.num_items
            n_clusters_items = self.n_clusters_items
        else:
            num_items = len(self.item_clustering)
            n_clusters_items = len(np.unique(item_clustering))
        
        user_clusters = sparse.csr_matrix(
            (np.ones(num_users),
            (range(num_users),
            user_clustering)),
            shape=(num_users, n_clusters_users))

        item_clusters = sparse.csr_matrix(
            (np.ones(num_items),
            (range(num_items),
            item_clustering)),
            shape=(num_items, n_clusters_items))

        mhk = user_clusters.T @ self.Y @ item_clusters
        return mhk
    
    ###########
    # compute yuk matrix using fast sparse matrix multiplication 
    def compute_yuk(self):
        item_clusters = sparse.csr_matrix(
            (np.ones(self.num_items),
            (range(self.num_items),
            self.item_clustering)),
            shape=(self.num_items, self.n_clusters_items))
        
        yuk = self.Y @ item_clusters
        return yuk
    
    ###########
    # compute yih matrix using fast sparse matrix multiplication 
    def compute_yih(self):
        user_clusters = sparse.csr_matrix(
            (np.ones(self.num_users),
            (range(self.num_users),
            self.user_clustering)),
            shape=(self.num_users, self.n_clusters_users))

        yih = self.Y.T @ user_clusters
        return yih
    
    ################
    # from list of covariates extracts the name, type and values
    def process_cov(self, cov_list):
        cov_names = []
        cov_types = []
        cov_values = []
        for cov in cov_list:
            cov_name, cov_type = cov[0].split('_')
            cov_names.append(cov_name)
            cov_types.append(cov_type)
            cov_values.append(cov[1])             
        return cov_names, cov_types, cov_values
    
    #############
    # computes nch matrix (maybe can be made faster using sparse matrices?)    
    def compute_nch(self, cov_values, clustering, n_clusters):
        cov_nch = []
        for cov in range(len(cov_values)):
            uniques = np.unique(cov_values[cov])
            nch = np.zeros((len(uniques), n_clusters))
            for h in range(n_clusters):
                mask = (clustering==h)
                for c in uniques:
                    nch[c, h] = (cov_values[cov][mask]==c).sum()    
            cov_nch.append(nch)
        return cov_nch
    
    ############
    # man comutation
    def gibbs_step(self):
        # do nothing for baseline
        pass
    
    ############
    # loop for fitting the model
    def gibbs_train(self, n_iters, verbose=0):
        np.random.seed(self.seed)
        
        self.n_iters = n_iters
        
        ll = compute_log_likelihood(nh = self.frequencies_users, nk = self.frequencies_items, a = self.prior_a, 
                                         b = self.prior_b, eps = self.epsilon, mhk=self.compute_mhk(), 
                                         user_clustering=self.user_clustering, 
                                        item_clustering=self.item_clustering,
                                        dg_u=np.zeros(self.num_users), 
                                        dg_i=np.zeros(self.num_items), 
                                        dg_cl_i=np.zeros(self.n_clusters_items), 
                                        dg_cl_u=np.zeros(self.n_clusters_users),
                                        degree_corrected=False)
        
        print('starting log likelihood', ll)
        llks = np.zeros(n_iters+1)
        user_cluster_list = np.zeros((n_iters+1, self.num_users), dtype=np.int32)
        item_cluster_list = np.zeros((n_iters+1, self.num_items), dtype=np.int32)
        frequencies_users_list = []
        frequencies_items_list = []
        
        llks[0] = ll
        user_cluster_list[0] = self.user_clustering.copy()
        item_cluster_list[0] = self.item_clustering.copy()
        frequencies_users_list.append(self.frequencies_users.copy())
        frequencies_items_list.append(self.frequencies_items.copy())
        
        for it in range(n_iters):
    
            self.gibbs_step()
            ll = compute_log_likelihood(nh = self.frequencies_users, nk = self.frequencies_items,a = self.prior_a, 
                                        b = self.prior_b, eps = self.epsilon,mhk = self.compute_mhk(),
                                        user_clustering=self.user_clustering, 
                                        item_clustering=self.item_clustering,
                                        dg_u=np.zeros(self.num_users), 
                                        dg_i=np.zeros(self.num_items), 
                                        dg_cl_i=np.zeros(self.n_clusters_items), 
                                        dg_cl_u=np.zeros(self.n_clusters_users), 
                                        degree_corrected=False)
            llks[it+1] += ll
            user_cluster_list[it+1] += self.user_clustering
            item_cluster_list[it+1] += self.item_clustering
            frequencies_users_list.append(self.frequencies_users.copy())
            frequencies_items_list.append(self.frequencies_items.copy())
            
            if verbose >= 1:
                if it % (n_iters // 10) == 0:
                    print(it, llks[it+1])
                if verbose >= 2:
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
        
        return llks, user_cluster_list, item_cluster_list
    
    ###########
    # estimate cluster assignment using the mode 
    # (i.e. most visited cluser for each user, item across samples)
    def estimate_cluster_assignment_mode(self, burn_in = 0, thinning = 1):
        if self.mcmc_draws_users is None:
            raise Exception('model must be trained first')
        
        assignment_users = -np.ones(self.num_users, dtype=np.int64)
        for u in range(self.num_users):
            assignment_users[u] = int(mode(self.mcmc_draws_users[burn_in::thinning, u])[0])
        
        assignment_items = -np.ones(self.num_items, dtype=np.int64)
        for i in range(self.num_items):
            assignment_items[i] = int(mode(self.mcmc_draws_items[burn_in::thinning, i])[0])
        
        self.user_clustering[:] = assignment_users
        _, frequencies_users = np.unique(assignment_users, return_counts=True)
        self.frequencies_users = frequencies_users
        
        self.item_clustering[:] = assignment_items
        _, frequencies_items = np.unique(assignment_items, return_counts=True)
        self.frequencies_items = frequencies_items
        
        # store estimation method
        self.estimated_items = 'mode'
        self.estimated_users = 'mode'
        
        return assignment_users, assignment_items
    
    ###########
    # estimate clister assignment minimizing the variation of information
    # (uses greedy algorithm as described in Wade and Ghahramani (2018))
    def estimate_cluster_assignment_vi(self, method='avg', max_k=None, burn_in=0, thinning=1):
        if method not in ['avg', 'comp', 'all']:
            raise Exception('invalid method')
        
        cc_users, cc_items = self.compute_co_clustering_matrix(burn_in=burn_in, thinning=thinning)
        
        psm_users = cc_users/np.max(cc_users)
        psm_items = cc_items/np.max(cc_items)

        res_users = minVI(psm_users, cls_draw = self.mcmc_draws_users[burn_in::thinning], method=method, max_k=max_k)
        res_items = minVI(psm_items, cls_draw = self.mcmc_draws_items[burn_in::thinning], method=method, max_k=max_k)
        
        est_cluster_users = res_users['cl']
        est_cluster_items = res_items['cl']
        
        # store the minimum vi
        vi_value_users = res_users['value']
        vi_value_items = res_items['value']
        
        self.user_clustering[:] = est_cluster_users
        unique_users, frequencies_users = np.unique(est_cluster_users, return_counts=True)
        self.frequencies_users = frequencies_users
        self.n_clusters_users = len(unique_users)
        
        self.item_clustering[:] = est_cluster_items
        unique_items, frequencies_items = np.unique(est_cluster_items, return_counts=True)
        self.frequencies_items = frequencies_items
        self.n_clusters_items = len(unique_items)
        
        # store which cl assignment method
        self.estimated_items = 'vi'
        self.estimated_users = 'vi'
                
        return est_cluster_users, est_cluster_items, vi_value_users, vi_value_items
    
    ##########
    # aux function to call the optimised function on relevant sample
    def compute_co_clustering_matrix(self, burn_in=0, thinning=1):
        if self.mcmc_draws_users is None:
            raise Exception('model must be trained first')
        
        cc_users = compute_co_clustering_matrix(self.mcmc_draws_users[burn_in::thinning])
        self.co_clustering_users = cc_users
        
        cc_items = compute_co_clustering_matrix(self.mcmc_draws_items[burn_in::thinning])
        self.co_clustering_items = cc_items
        
        return cc_users, cc_items
    
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
                llk_out.append(self.Y[u,i]*np.log(theta[zu, qi])-theta[zu,qi]-np.log(math.factorial(self.Y[u,i])))
        return llk_out
    
    ##########
    # generate point predictions from estimated parameters
    # takes as input pairs of user-item and outputs predicted rating
    def point_predict(self, pairs, seed=None):
        if seed is None:
            np.random.seed(self.seed)
        elif seed == -1:
            pass
        else:
            np.random.seed(seed)
        
        uniques_ratings, frequencies_ratings = np.unique(self.Y, return_counts=True)
        
        preds = []
        for u, i in pairs:
            # predict with predictive posterior mean
            preds.append(np.random.choice(uniques_ratings, p=frequencies_ratings/np.sum(frequencies_ratings)))

        return preds
    
    ##################
    # for a list of users returns unconsumed items in the cluster with highest score
    def predict_with_ranking(self, users):        
        top_cluster =[]
        for u in users:
            num = np.random.randint(1, self.num_items)
            choice = np.random.choice(self.num_items, num, replace=False)
            top_cluster.append(choice)

        return top_cluster
    
    ############
    # for a given list of users returns k suggested items
    def predict_k(self, users, k):        
        out = []
        for u in users:
            choice = np.random.choice(self.num_items, k, replace=False)
            out.append(choice)

        return out
    