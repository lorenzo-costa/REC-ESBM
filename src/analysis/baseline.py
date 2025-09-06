import numpy as np
from scipy import sparse
from scipy.stats import mode

from analysis.numba_functions import sampling_scheme, compute_log_probs_cov, compute_log_likelihood
from vi_functs import minVI
from analysis.numba_functions import compute_co_clustering_matrix


#########################################
# baseline class
########################################
class Baseline:
    """Baseline ESBM model

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
    verbose_users : bool
        whether to print verbose output for user-related computations, by default False
    verbose_items : bool
        whether to print verbose output for item-related computations, by default False
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

        # a lot of type and value checking
        if num_items is None or not isinstance(num_items, int) or num_items <= 0:
            raise Exception('please provide valid number of items')
        
        if num_users is None or not isinstance(num_users, int) or num_users <= 0:
            raise Exception('please provide valid number of users')
        
        if prior_a <= 0 or not isinstance(prior_a, (int, float)):
            raise Exception('please provide valid prior a parameter (>0)')
        if prior_b <= 0 or not isinstance(prior_b, (int, float)):
            raise Exception('please provide valid prior b parameter (>0)')

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
        
        self.degree_param_users = degree_param_users
        self.degree_param_items = degree_param_items
        
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
                        
        if cov_users is not None:
            self.cov_names_users, self.cov_types_users, self.cov_values_users = self.process_cov(cov_users)
    
        if cov_items is not None:
            self.cov_names_items, self.cov_types_items, self.cov_values_items = self.process_cov(cov_items)
        
        # clustering structure not provided create it
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
        
        # theta not provided generate it
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
            
        # if adjancecy matrix not given generate it
        if Y is None:
            print('randomly initialising data')
            self.generate_data()
        else:
            self.Y = Y
        

    def process_clusters(self, user_clustering, item_clustering):
        """Computes cluster metrics.

        Parameters
        ----------
        user_clustering : list or array-like
            Cluster assignments for users.
        item_clustering : list or array-like
            Cluster assignments for items.
        """
        
        occupied_clusters_users, frequencies_users = np.unique(user_clustering, return_counts=True)
        self.frequencies_users = frequencies_users
        self.n_clusters_users = len(occupied_clusters_users)
        self.user_clustering = np.array(user_clustering)   
                    
        occupied_clusters_items, frequencies_items = np.unique(item_clustering, return_counts=True)
        self.frequencies_items = frequencies_items
        self.n_clusters_items = len(occupied_clusters_items)
        self.item_clustering = np.array(item_clustering)
    
    
    def generate_data(self):
        """Generates random data according to the model.
        """
        
        np.random.seed(self.seed)
        # Y params collects the appropriate theta entries accoridng to cluster structure
        Y_params = self.theta[self.user_clustering][:, self.item_clustering]
        Y = np.random.poisson(Y_params)
        self.Y = Y.copy()
        return
    
    def init_cluster_random(self, user_clustering=None, item_clustering=None):
        """Initialises random clustering structure according to the prior.

        Parameters
        ----------
        user_clustering : list, optional
            Initial user clustering, by default None
        item_clustering : list, optional
            Initial item clustering, by default None

        Returns
        -------
        user_clustering : list
            Final user clustering.
        item_clustering : list
            Final item clustering.
        """
        
        np.random.seed(self.seed)
        # note this if is used becuase we may have only one of the two being random
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
                # prior contribution
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
                    # make new cluster
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
                    # make new cluster
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
    
    def compute_mhk(self, user_clustering=None, item_clustering=None):
        """Computes the MHK matrix using (fast) sparse matrix multiplication.

        Parameters
        ----------
        user_clustering : list, optional
            Clustering of users, by default None
        item_clustering : list, optional
            Clustering of items, by default None

        Returns
        -------
        mhk : np.array
            MHK matrix
        """
        
        if user_clustering is None:
            user_clustering = self.user_clustering
            num_users = self.num_users
            n_clusters_users = self.n_clusters_users
        else:
            num_users = len(user_clustering)
            n_clusters_users = len(np.unique(user_clustering))
            
        if item_clustering is None:
            item_clustering = self.item_clustering
            num_items = self.num_items
            n_clusters_items = self.n_clusters_items
        else:
            num_items = len(item_clustering)
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
     
    def compute_yuk(self):
        """Computes the YUK matrix.

        Returns
        -------
        yuk : np.array
            YUK matrix
        """
        item_clusters = sparse.csr_matrix(
            (np.ones(self.num_items),
            (range(self.num_items),
            self.item_clustering)),
            shape=(self.num_items, self.n_clusters_items))
        
        yuk = self.Y @ item_clusters
        return yuk
    
    def compute_yih(self):
        """Computes the YIH matrix.

        Returns
        -------
        yih : np.array
            YIH matrix
        """
        user_clusters = sparse.csr_matrix(
            (np.ones(self.num_users),
            (range(self.num_users),
            self.user_clustering)),
            shape=(self.num_users, self.n_clusters_users))

        yih = self.Y.T @ user_clusters
        return yih
    
    
    def process_cov(self, cov_list):
        """Processes a list of covariates.

        Parameters
        ----------
        cov_list : list of tuples
            list of tuples (covname_covtype, covvalues)

        Returns
        -------
        tuple: (cov_names, cov_types, cov_values)
            cov_names: list of covariate names
            cov_types: list of covariate types
            cov_values: list of covariate values
        """
        cov_names = []
        cov_types = []
        cov_values = []
        for cov in cov_list:
            cov_name, cov_type = cov[0].split('_')
            cov_names.append(cov_name)
            cov_types.append(cov_type)
            cov_values.append(cov[1])             
        return cov_names, cov_types, cov_values
     
    def compute_nch(self, cov_values, clustering, n_clusters):
        """Computes the NCH matrix.

        Parameters
        ----------
        cov_values : list
            list of covariate values
        clustering : list
            list of cluster assignments
        n_clusters : int
            number of clusters

        Returns
        -------
        list
            list of NCH matrices
        """
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
    
    
    def gibbs_step(self):
        """Performs a Gibbs sampling step.
        """
        # do nothing for baseline
        return
    
    def gibbs_train(self, n_iters, verbose=0):
        """Trains the model using Gibbs sampling.

        Parameters
        ----------
        n_iters : int
            Number of iterations for Gibbs sampling.
        verbose : int, optional
            Verbosity level, by default 0. 0: no output, 1: every 10% of iterations,
            2: also print frequencies, 3: also print cluster assignments

        Returns
        -------
        tuple: (llks, user_cluster_list, item_cluster_list)
            llks: log-likelihood values during training
            user_cluster_list: MCMC samples of user cluster assignments during training
            item_cluster_list: MCMC samples of item cluster assignments during training
        """
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
    
    
    def estimate_cluster_assignment_mode(self, burn_in = 0, thinning = 1):
        """Estimate cluster assignments using the mode.

        Parameters
        ----------
        burn_in : int, optional
            Number of initial samples to discard, by default 0
        thinning : int, optional
            Thinning factor for MCMC samples, by default 1

        Returns
        -------
        tuple (user_cluster_assignments, item_cluster_assignments)
            user_cluster_assignments : np.ndarray
                Estimated cluster assignments for users
            item_cluster_assignments : np.ndarray
                Estimated cluster assignments for items

        Raises
        ------
        Exception
            If the model has not been trained.
        """
        
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
    
    
    
    def estimate_cluster_assignment_vi(self, method='avg', max_k=None, burn_in=0, thinning=1):
        """Estimate cluster assignments minimizing the variation of information.
        
        Uses a greedy algorithm as described in Wade and Ghahramani (2018))

        Parameters
        ----------
        method : str, optional
            Estimation method to use, by default 'avg'
        max_k : int, optional
            Maximum number of clusters to consider for greedy optimization, by default int(np.ceil(psm.shape[0] / 8))
        burn_in : int, optional
            Number of initial samples to discard, by default 0
        thinning : int, optional
            Thinning factor for MCMC samples, by default 1

        Returns
        -------
        dict
            Dictionary containing the best cluster assignments and their corresponding VI value.

        Raises
        ------
        Exception
            If the model has not been trained.
        """
        
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
    
    
    
    def compute_co_clustering_matrix(self, burn_in=0, thinning=1):
        """Aux function to call the optimised function on relevant sample"""
        if self.mcmc_draws_users is None:
            raise Exception('model must be trained first')
        
        cc_users = compute_co_clustering_matrix(self.mcmc_draws_users[burn_in::thinning])
        self.co_clustering_users = cc_users
        
        cc_items = compute_co_clustering_matrix(self.mcmc_draws_items[burn_in::thinning])
        self.co_clustering_items = cc_items
        
        return cc_users, cc_items
    
    
    
    def point_predict(self, pairs, seed=None):
        """Predict ratings for user-item pairs.

        Parameters
        ----------
        pairs : list of tuples
            List of (user, item) pairs for which to predict ratings.
        seed : int, optional
            Random seed for reproducibility, by default None

        Returns
        -------
        preds : list
            List of predicted ratings corresponding to the input pairs.
        """
        
        if seed is None:
            np.random.seed(self.seed)
        elif seed == -1:
            pass
        else:
            np.random.seed(seed)
        
        uniques_ratings, frequencies_ratings = np.unique(self.Y, return_counts=True)
        
        preds = []
        for u, i in pairs:
            # baseline: predict with predictive posterior mean
            preds.append(np.random.choice(uniques_ratings, p=frequencies_ratings/np.sum(frequencies_ratings)))

        return preds
    
    
    def predict_with_ranking(self, users):
        """Predict items for users based on cluster with highest score.

        Parameters
        ----------
        users : list
            List of users for whom to predict items.

        Returns
        -------
        list
            List of predicted item indices for each user.
        """
        top_cluster =[]
        for u in users:
            # baseline: completely random
            num = np.random.randint(1, self.num_items)
            choice = np.random.choice(self.num_items, num, replace=False)
            top_cluster.append(choice)

        return top_cluster
    
    

    def predict_k(self, users, k):
        """Predict k items for each user based on cluster with highest score.

        Parameters
        ----------
        users : list
            List of users for whom to predict items.
        k : int
            Number of items to predict for each user.

        Returns
        -------
        list
            List of predicted item indices for each user.
        """
        out = []
        for u in users:
            # baseline: completely random
            choice = np.random.choice(self.num_items, k, replace=False)
            out.append(choice)

        return out
    