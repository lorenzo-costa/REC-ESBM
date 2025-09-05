from sklearn.metrics import mean_squared_error, mean_absolute_error
from aux_functions import compute_precision, compute_recall
from vi_functs import VI
import numpy as np


def validate_models(Y_train, 
                    Y_val, 
                    model_list, 
                    param_list,
                    relevant_threshold=2, 
                    n_iters=500, 
                    burn_in=None, 
                    thinning=3, 
                    verbose=0,
                    model_names=None, 
                    true_users=None, 
                    true_items=None, 
                    k=10,
                    print_intermid=False,
                    seed=42):    
    """Fits a list of models on training data and validates them using a validation dataset.
    
    This function trains each model in `model_list` using the provided training data (`Y_train`) and model-specific parameters (`param_list`).
    After training, it evaluates each model on the validation dataset (`Y_val`) using several metrics:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Precision@k and Recall@k (ranking metrics based on relevant items per user)
    Optionally, if the true user and item cluster assignments (`true_users`, `true_items`) are provided, the function computes the Variation of Information (VI) distance between the estimated and true partitions.
    
    Inputs
    ----------
    Y_train : array-like
        Training dataset containing user-item interactions.
    Y_val : array-like
        Validation dataset containing user-item interactions.
    model_list : list
        List of model classes to be trained and validated.
    param_list : list
        List of dictionaries containing parameters for each model.
    relevant_threshold : float, optional
        Threshold for considering an item relevant (default is 2).
    n_iters : int, optional
        Number of gibbs iterations for model training (default is 500).
    burn_in : int, optional
        Number of burn-in iterations for inference (default is n_iters//2).
    thinning : int, optional
        Thinning interval for inference (default is 3).
    k : int, optional
        Number of top items to consider for ranking metrics (default is 10).
        If None returns all items in the block with highest theta.
    model_names : list, optional
        List of names for each model (default is None).
    true_users : array-like, optional
        True user cluster assignments for VI computation (default is None).
    true_items : array-like, optional
        True item cluster assignments for VI computation (default is None).
    print_intermid : bool, optional
        If True, prints intermediate results for each model (default is False).
    verbose : int, optional
        Verbosity level for gibbs training routine (default is 0).
    
    Outputs
    -------
    model_list_out : list
        List of trained model instances, each annotated with validation metrics (MAE, MSE, precision@k, recall@k, VI distances if applicable).
    """

    if burn_in is None:
        burn_in = n_iters//2
    
    # prepare val set
    Y_val_pairs = [(u,i) for u,i,_ in Y_val]
    Y_val_users = [u for u,_,_ in Y_val]
    Y_val_items = [i for _,i,_ in Y_val]
    Y_val_ratings = [r for _,_,r in Y_val]

    # find which items are relevant (i.e. rating > relevant_threshold) for each user
    val_users_relevant = {}
    for j in range(len(Y_val_pairs)):
        u = Y_val_users[j]
        i = Y_val_items[j]
        r = Y_val_ratings[j]
        if u not in val_users_relevant:
            val_users_relevant[u] = [] 
        if r >= relevant_threshold:
            val_users_relevant[u].append(i)
    val_users_unique = list(val_users_relevant.keys())
    
    # model validations
    model_list_out = []
    for i in range(len(model_list)):
        if model_names is not None:
            name = model_names[i]
        else:
            name = i
            
        print('\nModel name:', name)
        model_type = model_list[i]
        params = param_list[i]
        model = model_type(Y=Y_train, num_users=Y_train.shape[0], num_items=Y_train.shape[1], **params)

        print('Starting training for model', name)
        _ = model.gibbs_train(n_iters, verbose=verbose)
        _ = model.estimate_cluster_assignment_vi(burn_in=burn_in, thinning=thinning)
        
        print('Starting prediction for model', name)
        model_ratings = model.point_predict(Y_val_pairs, seed=seed)
        mae_model = mean_absolute_error(Y_val_ratings, model_ratings)
        mse_model = mean_squared_error(Y_val_ratings, model_ratings)    
        
        print('Starting ranking for model', name)
        if k is None:
            ranks_model = model.predict_with_ranking(val_users_unique)
        else:
            ranks_model = model.predict_k(val_users_unique, k=k)
            
        precision_list_model = []   
        recall_list_model = []
        for j in range(len(val_users_unique)):
            if len(val_users_relevant[val_users_unique[j]]) == 0:
                # no relevant items for that user
                continue
            precision_list_model.append(compute_precision(val_users_relevant[val_users_unique[j]], ranks_model[j]))
            recall_list_model.append(compute_recall(val_users_relevant[val_users_unique[j]], ranks_model[j]))

        if precision_list_model == [] or recall_list_model == []:
            raise Exception('No users with relevant items in the validation set. Try changing the relevant_threshold parameter.')
        
        precision_model = sum(precision_list_model)/len(precision_list_model)
        recall_model = sum(recall_list_model)/len(recall_list_model)

        if true_users is not None:
            vi_users_model = VI(true_users, model.user_clustering)[0]
            model.vi_users = vi_users_model
        if true_items is not None:
            vi_items_model = VI(true_items, model.item_clustering)[0]
            model.vi_items = vi_items_model
        
        model.precision = precision_model
        model.recall = recall_model
        model.mae = mae_model
        model.mse = mse_model
        
        if print_intermid is True:
            print('MAE:', mae_model)
            print('MSE:', mse_model)
            print('Precision:', precision_model)
            print('Recall:', recall_model)
            if true_users is not None:
                print('VI users:', vi_users_model)
            if true_items is not None:
                print('VI items:', vi_items_model)
        
        model_list_out.append(model)
        
    return model_list_out


def generate_val_set(y, 
                     size=0.1, 
                     seed=42, 
                     only_observed=True):
    
    np.random.seed(seed)
    n_users, n_items = y.shape
    n_val = int(size*n_users*n_items)
    y_val = []
    for _ in range(n_val):
        u = np.random.randint(n_users)
        i = np.random.randint(n_items)
        if only_observed:
            while y[u,i] == 0:
                u = np.random.randint(n_users)
                i = np.random.randint(n_items)
        y_val.append((u,i, int(y[u,i])))
    
    y_train = y.copy()
    for u,i, _ in y_val:
        y_train[u,i] = 0
    
    return y_train, y_val


def multiple_runs(true_mod,
                  num_users, 
                  num_items, 
                  n_cl_u, 
                  n_cl_i, 
                  n_runs, 
                  n_iters, 
                  params_list, 
                  model_list, 
                  model_names=None,
                  cov_places_items=None, 
                  cov_places_users=None,
                  k = 10, 
                  print_intermid=True, 
                  verbose=1, 
                  burn_in=0, 
                  thinning=1, 
                  seed=0, 
                  params_init=None):
    
    """
    Runs multiple experiments by generating synthetic data from a specified model and evaluating a list of models.
    For each run, the function:
    - Randomly assigns users and items to clusters.
    - Generates covariates for users and items based on their cluster assignments.
    - Initializes the true model using the provided parameters and cluster assignments.
    - Splits the generated data into training and validation sets.
    - Optionally updates covariates in the parameter list for specific models.
    - Evaluates each model in `model_list` using the `validate_models` subroutine.
    - Collects evaluation metrics (MSE, MAE, precision, recall, VI for users/items) for each model.
    
    Inputs
    ----------
    true_mod : callable
        The model class or function used to generate synthetic data.
    num_users : int
        Number of users in the synthetic dataset.
    num_items : int
        Number of items in the synthetic dataset.
    n_cl_u : int
        Number of clusters for users.
    n_cl_i : int
        Number of clusters for items.
    n_runs : int
        Number of experimental runs to perform.
    n_iters : int
        Number of gibbs step for each model validation.
    params_list : list
        List of parameter dictionaries for each model to be validated.
    model_list : list
        List of model classes.
    model_names : list, optional
        List of names corresponding to each model.
    cov_places_items : list, optional
        Indices in `params_list` to update item covariates.
    cov_places_users : list, optional
        Indices in `params_list` to update user covariates.
    k : int, optional
        Number of top items to consider for precision@k/recall@k metrics (default is 10).
    print_intermid : bool, optional
        Whether to print intermediate results. (default is True).
    verbose : int, optional
        Verbosity level for model validation. (default is 1).
    burn_in : int, optional
        Number of burn-in iterations for model validation. (default is 0).
    thinning : int, optional
        Thinning interval for model validation. (default is 1).
    seed : int, optional
        Random seed for reproducibility. (default is 0).
    params_init : dict, optional
        Initial parameters for the true model.

    Outputs
    -------
    names_list : list
        Names of models evaluated.
    mse_list : list
        Mean squared error for each model/run.
    mae_list : list
        Mean absolute error for each model/run.
    precision_list : list
        Precision@k for each model/run.
    recall_list : list
        Recall@k for each model/run.
    vi_users_list : list
        Variation of information for user clustering for each model/run.
    vi_items_list : list
        Variation of information for item clustering for each model/run.
    models_list_out : list
        List of model outputs for each model/run.
    """
                  
    names_list = []
    models_list_out = []
    mse_list = []
    mae_list = []
    precision_list = []
    recall_list = []
    vi_users_list = []
    vi_items_list = []
    
    for r in range(n_runs):
        np.random.seed(seed+r)
        places = np.round(np.random.dirichlet(np.ones(n_cl_u))*num_users)
        
        # test
        if sum(places) != num_users:
            places[0] += (num_users - sum(places))
        
        user_clustering = []
        for i in range(len(places)):
            # using >= (instead of >) here because otherwise it may include higher numbers before lower number
            # e.g. a clustering like 0,0,2,2,3 if 1 is not present it messes up stuff
            while len(user_clustering) < num_users and places[i]>=0: 
                user_clustering.append(i)
                places[i]-=1
        
        temp = np.array([1 if user_clustering[i]%2==0 else 0 for i in range(num_users)])
        cov_users = [('cov1_cat', temp.copy())]
        
        places = np.round(np.random.dirichlet(np.ones(n_cl_i))*num_items)
        
        #test
        if sum(places) != num_items:
            places[0] += (num_items - sum(places))
        
        item_clustering = []
        for i in range(len(places)):
            while len(item_clustering) < num_items and places[i]>=0:
                item_clustering.append(i)
                places[i]-=1
        
        temp = np.array([1 if item_clustering[i]%2==0 else 0 for i in range(num_items)])
        cov_items = [('cov1_cat', temp.copy())]
        
        if params_init is None:
            params_init = {}
            params_init['user_clustering'] = user_clustering
            params_init['item_clustering'] = item_clustering
            params_init['cov_users'] = cov_users
            params_init['cov_items'] = cov_items        
            params_init['seed'] = seed+r
        
        true_model = true_mod(**params_init)
        
        Y_train, Y_val = generate_val_set(true_model.Y, size=0.1, seed=42, only_observed=False)
        true_users = true_model.user_clustering.copy()
        true_items = true_model.item_clustering.copy()
        
        if cov_places_users is not None:
            for place in cov_places_users:
                params_list[place]['cov_users'] = cov_users
        if cov_places_items is not None:
            for place in cov_places_items:
                params_list[place]['cov_items'] = cov_items
        
        out = validate_models(Y_train, Y_val, model_list, params_list, n_iters=n_iters, burn_in=burn_in, k = k, 
                              verbose=verbose, thinning=thinning, model_names=model_names, true_users=true_users, 
                              true_items=true_items, print_intermid=print_intermid)
        
        for m in range(len(out)):
            names_list.append(model_names[m])
            mse_list.append(out[m].mse)
            mae_list.append(out[m].mae)
            precision_list.append(out[m].precision)
            recall_list.append(out[m].recall)
            vi_users_list.append(out[m].vi_users)
            vi_items_list.append(out[m].vi_items)
            models_list_out.append(out[m]) 
    
    return names_list, mse_list, mae_list, precision_list, recall_list, vi_users_list, vi_items_list, models_list_out