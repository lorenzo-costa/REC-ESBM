from sklearn.metrics import mean_squared_error, mean_absolute_error
from aux_functions import compute_precision, compute_recall
from vi_functs import VI
import numpy as np

def validate_models(Y_train, Y_val, model_list, param_list, n_iters=500, burn_in=None, verbose=0,
                    thinning=3, model_names=None, true_users=None, true_items=None, k=None,
                    print_intermid=False):
    if burn_in is None:
        burn_in = n_iters//2
    
    Y_val_pairs = [(u,i) for u,i,_ in Y_val]
    Y_val_users = [u for u,_,_ in Y_val]
    Y_val_items = [i for _,i,_ in Y_val]
    Y_val_ratings = [r for _,_,r in Y_val]
    
    
    val_users_relevant = {}

    for j in range(len(Y_val_pairs)):
        u = Y_val_users[j]
        i = Y_val_items[j]
        r = Y_val_ratings[j]
        if u not in val_users_relevant:
            val_users_relevant[u] = [] 
        if r > 1:
            val_users_relevant[u].append(i)
    
    val_users_unique = list(val_users_relevant.keys())
    
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
        llk_model, user_cl_model, item_cl_model = model.gibbs_train(n_iters, verbose=verbose)
        model_est_users, model_est_items, model_vi_users, model_vi_items = model.estimate_cluster_assignment_vi(burn_in=burn_in, thinning=thinning)
        
        waic_model = None
        
        
        print('Starting waic computation', name)        
        llk_edges = []
        # for iter in range(burn_in, model.n_iters, thinning):
        #     llk_edges.append(model.compute_llk(iter))
        # waic_model = waic_calculation(np.array(llk_edges))
        
        print('Starting prediction for model', name)
        model_ratings = model.point_predict(Y_val_pairs, seed=42)
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
                continue
            precision_list_model.append(compute_precision(val_users_relevant[val_users_unique[j]], ranks_model[j]))
            recall_list_model.append(compute_recall(val_users_relevant[val_users_unique[j]], ranks_model[j]))
        precision_model = sum(precision_list_model)/len(precision_list_model)
        recall_model = sum(recall_list_model)/len(recall_list_model)
        
        if true_users is not None:
            vi_users_model = VI(true_users, model.user_clustering)[0]
            model.vi_users = vi_users_model
        if true_items is not None:
            vi_items_model = VI(true_items, model.item_clustering)[0]
            model.vi_items = vi_items_model
        
        model.precision_ranks = precision_model
        model.recall_ranks = recall_model
        model.mae = mae_model
        model.mse = mse_model
        model.waic = waic_model
        model.llk_edges = llk_edges
        
        if print_intermid is True:
            print('MAE:', mae_model)
            print('MSE:', mse_model)
            print('Precision:', precision_model)
            print('Recall:', recall_model)
            if true_users is not None:
                print('VI users:', vi_users_model)
            if true_items is not None:
                print('VI items:', vi_items_model)
            if waic_model is not None:
                print('WAIC:', waic_model)
        
        model_list_out.append(model)
        
    return model_list_out


def generate_val_set(y, size=0.1, seed=42, only_observed=True):
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


def multiple_runs(true_mod, num_users, num_items, n_cl_u, n_cl_i, n_runs, 
                  n_iters, params_list, model_list, model_names,
                  cov_places_items=None, cov_places_users=None,
                  k = 10, print_intermid=True, verbose=1, 
                  burn_in=0, thinning=1, seed=0, params_init=None):
    
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
        user_clustering = list(np.random.choice(np.arange(n_cl_u), p=[0.6, 0.2, 0.15, 0.05],size=num_users))
        t = np.array([1 if user_clustering[i]%2==0 else 0 for i in range(num_users)])
        t[np.random.randint(0, len(t), size=25)] = 0
        cov_users = [('gender_cat', t.copy())]

        item_clustering = list(np.random.choice(np.arange(n_cl_i), p=[0.5, 0.25, 0.2, 0.05],size=num_items))
        t3 = np.array([1 if item_clustering[i]%2==0 else 0 for i in range(num_items)])
        t3[np.random.randint(0, len(t3), size=25)] = 0
        cov_items = [('genre_cat', t3.copy()) for _ in range(1)]
        
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
            precision_list.append(out[m].precision_ranks)
            recall_list.append(out[m].recall_ranks)
            vi_users_list.append(out[m].vi_users)
            vi_items_list.append(out[m].vi_items)
            models_list_out.append(out[m])
            
    
    return names_list, mse_list, mae_list, precision_list, recall_list, vi_users_list, vi_items_list, models_list_out