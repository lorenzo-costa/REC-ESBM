import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aux_functions import plot_heatmap
from esbm_rec import esbm
from dc_esbm_rec import dcesbm
from valid_functs import multiple_runs
import yaml

with open("src/analysis/config_sim.yaml", "r") as f:
    config = yaml.safe_load(f)

n_users = config["general_params"]["num_users"]
n_items = config["general_params"]["num_items"]
n_cl_u = config["general_params"]["bar_h_users"]
n_cl_i = config["general_params"]["bar_h_items"]
n_iters = config["run_settings"]["num_iters"]
burn_in = config["run_settings"]["burn_in"]
thinning = config["run_settings"]["thinning"]
k = config["run_settings"]["k"]
n_runs = config["run_settings"]["n_runs"]
seed = config["run_settings"]["seed"]

params_baseline = config["params_baseline"]
params_dp = config["params_variants"]["dp"]
params_py = config["params_variants"]["py"]
params_gn = config["params_variants"]["gn"]
params_dp_cov = config["params_variants"]["dp_cov"]
params_py_cov = config["params_variants"]["py_cov"]
params_gn_cov = config["params_variants"]["gn_cov"]
params_init = config["params_init"]

cov_places_users = [3,4,5]
cov_places_items = [3,4,5]

model_list = [dcesbm, dcesbm, dcesbm, dcesbm, dcesbm, dcesbm, 
              esbm, esbm, esbm, esbm, esbm, esbm]
params_list = [params_dp, params_py, params_gn, params_dp_cov, params_py_cov, params_gn_cov,
               params_dp, params_py, params_gn, params_dp_cov, params_py_cov, params_gn_cov]

model_names = ['dc_DP', 'dc_PY', 'dc_GN', 'dc_DP_cov', 'dc_PY_cov', 'dc_GN_cov',
               'esbm_DP', 'esbm_PY', 'esbm_GN', 'esbm_DP_cov', 'esbm_PY_cov', 'esbm_GN_cov']

out = multiple_runs(true_mod=dcesbm, 
                    params_init=params_init, 
                    num_users=n_users, 
                    num_items=n_items, 
                    n_cl_u=n_cl_u, 
                    n_cl_i=n_cl_i, 
                    n_runs=n_runs, 
                    n_iters=n_iters,
                    params_list=params_list, 
                    model_list=model_list, 
                    model_names=model_names, 
                    cov_places_users=cov_places_users, 
                    cov_places_items=cov_places_items, 
                    k=k, 
                    print_intermid=True, 
                    verbose=1, 
                    burn_in=burn_in, 
                    thinning=thinning, 
                    seed=seed) 

names_list, mse_list, mae_list, precision_list, recall_list, vi_users_list, vi_items_list, models_list_out = out

dc_mean_dp_mse = np.mean(mse_list[0::12])
dc_mean_py_mse = np.mean(mse_list[1::12])
dc_mean_gn_mse = np.mean(mse_list[2::12])
dc_mean_dp_cov_mse = np.mean(mse_list[3::12])
dc_mean_py_cov_mse = np.mean(mse_list[4::12])
dc_mean_gn_cov_mse = np.mean(mse_list[5::12])

dc_mean_dp_mae = np.mean(mae_list[0::12])
dc_mean_py_mae = np.mean(mae_list[1::12])
dc_mean_gn_mae = np.mean(mae_list[2::12])
dc_mean_dp_cov_mae = np.mean(mae_list[3::12])
dc_mean_py_cov_mae = np.mean(mae_list[4::12])
dc_mean_gn_cov_mae = np.mean(mae_list[5::12])

dc_mean_dp_prec = np.mean(precision_list[0::12])
dc_mean_py_prec = np.mean(precision_list[1::12])
dc_mean_gn_prec = np.mean(precision_list[2::12])
dc_mean_dp_cov_prec = np.mean(precision_list[3::12])
dc_mean_py_cov_prec = np.mean(precision_list[4::12])
dc_mean_gn_cov_prec = np.mean(precision_list[5::12])

dc_mean_dp_rec = np.mean(recall_list[0::12])
dc_mean_py_rec = np.mean(recall_list[1::12])
dc_mean_gn_rec = np.mean(recall_list[2::12])
dc_mean_dp_cov_rec = np.mean(recall_list[3::12])
dc_mean_py_cov_rec = np.mean(recall_list[4::12])
dc_mean_gn_cov_rec = np.mean(recall_list[5::12])

dc_mean_dp_vi_users = np.mean(vi_users_list[0::12])
dc_mean_py_vi_users = np.mean(vi_users_list[1::12])
dc_mean_gn_vi_users = np.mean(vi_users_list[2::12])
dc_mean_dp_cov_vi_users = np.mean(vi_users_list[3::12])
dc_mean_py_cov_vi_users = np.mean(vi_users_list[4::12])
dc_mean_gn_cov_vi_users = np.mean(vi_users_list[5::12])

dc_mean_dp_vi_items = np.mean(vi_items_list[0::12])
dc_mean_py_vi_items = np.mean(vi_items_list[1::12])
dc_mean_gn_vi_items = np.mean(vi_items_list[2::12])
dc_mean_dp_cov_vi_items = np.mean(vi_items_list[3::12])
dc_mean_py_cov_vi_items = np.mean(vi_items_list[4::12])
dc_mean_gn_cov_vi_items = np.mean(vi_items_list[5::12])

esbm_mean_dp_mse = np.mean(mse_list[6::12])
esbm_mean_py_mse = np.mean(mse_list[7::12])
esbm_mean_gn_mse = np.mean(mse_list[8::12])
esbm_mean_dp_cov_mse = np.mean(mse_list[9::12])
esbm_mean_py_cov_mse = np.mean(mse_list[10::12])
esbm_mean_gn_cov_mse = np.mean(mse_list[11::12])    

esbm_mean_dp_mae = np.mean(mae_list[6::12])
esbm_mean_py_mae = np.mean(mae_list[7::12])
esbm_mean_gn_mae = np.mean(mae_list[8::12])
esbm_mean_dp_cov_mae = np.mean(mae_list[9::12])
esbm_mean_py_cov_mae = np.mean(mae_list[10::12])
esbm_mean_gn_cov_mae = np.mean(mae_list[11::12])

esbm_mean_dp_prec = np.mean(precision_list[6::12])
esbm_mean_py_prec = np.mean(precision_list[7::12])
esbm_mean_gn_prec = np.mean(precision_list[8::12])
esbm_mean_dp_cov_prec = np.mean(precision_list[9::12])
esbm_mean_py_cov_prec = np.mean(precision_list[10::12])
esbm_mean_gn_cov_prec = np.mean(precision_list[11::12])

esbm_mean_dp_rec = np.mean(recall_list[6::12])
esbm_mean_py_rec = np.mean(recall_list[7::12])
esbm_mean_gn_rec = np.mean(recall_list[8::12])
esbm_mean_dp_cov_rec = np.mean(recall_list[9::12])
esbm_mean_py_cov_rec = np.mean(recall_list[10::12])
esbm_mean_gn_cov_rec = np.mean(recall_list[11::12])

esbm_mean_dp_vi_users = np.mean(vi_users_list[6::12])
esbm_mean_py_vi_users = np.mean(vi_users_list[7::12])
esbm_mean_gn_vi_users = np.mean(vi_users_list[8::12])
esbm_mean_dp_cov_vi_users = np.mean(vi_users_list[9::12])
esbm_mean_py_cov_vi_users = np.mean(vi_users_list[10::12])
esbm_mean_gn_cov_vi_users = np.mean(vi_users_list[11::12])

esbm_mean_dp_vi_items = np.mean(vi_items_list[6::12])
esbm_mean_py_vi_items = np.mean(vi_items_list[7::12])
esbm_mean_gn_vi_items = np.mean(vi_items_list[8::12])
esbm_mean_dp_cov_vi_items = np.mean(vi_items_list[9::12])
esbm_mean_py_cov_vi_items = np.mean(vi_items_list[10::12])
esbm_mean_gn_cov_vi_items = np.mean(vi_items_list[11::12])


output_table = pd.DataFrame()

output_table['Model'] = ['dc_DP', 'dc_PY', 'dc_GN', 'dc_DP_cov', 'dc_PY_cov', 'dc_GN_cov',
                         'esbm_DP', 'esbm_PY', 'esbm_GN', 'esbm_DP_cov', 'esbm_PY_cov', 'esbm_GN_cov']
output_table['MAE'] = [dc_mean_dp_mae, dc_mean_py_mae, dc_mean_gn_mae, dc_mean_dp_cov_mae, dc_mean_py_cov_mae, dc_mean_gn_cov_mae,
                       esbm_mean_dp_mae, esbm_mean_py_mae, esbm_mean_gn_mae, esbm_mean_dp_cov_mae, esbm_mean_py_cov_mae, esbm_mean_gn_cov_mae]

output_table['MSE'] = [dc_mean_dp_mse, dc_mean_py_mse, dc_mean_gn_mse, dc_mean_dp_cov_mse, dc_mean_py_cov_mse, dc_mean_gn_cov_mse,
                       esbm_mean_dp_mse, esbm_mean_py_mse, esbm_mean_gn_mse, esbm_mean_dp_cov_mse, esbm_mean_py_cov_mse, esbm_mean_gn_cov_mse]

output_table['Precision'] = [dc_mean_dp_prec, dc_mean_py_prec, dc_mean_gn_prec, dc_mean_dp_cov_prec, dc_mean_py_cov_prec, dc_mean_gn_cov_prec,
                             esbm_mean_dp_prec, esbm_mean_py_prec, esbm_mean_gn_prec, esbm_mean_dp_cov_prec, esbm_mean_py_cov_prec, esbm_mean_gn_cov_prec]

output_table['Recall'] = [dc_mean_dp_rec, dc_mean_py_rec, dc_mean_gn_rec, dc_mean_dp_cov_rec, dc_mean_py_cov_rec, dc_mean_gn_cov_rec,
                          esbm_mean_dp_rec, esbm_mean_py_rec, esbm_mean_gn_rec, esbm_mean_dp_cov_rec, esbm_mean_py_cov_rec, esbm_mean_gn_cov_rec]

output_table['VI_users'] = [dc_mean_dp_vi_users, dc_mean_py_vi_users, dc_mean_gn_vi_users, dc_mean_dp_cov_vi_users, dc_mean_py_cov_vi_users, dc_mean_gn_cov_vi_users,
                           esbm_mean_dp_vi_users, esbm_mean_py_vi_users, esbm_mean_gn_vi_users, esbm_mean_dp_cov_vi_users, esbm_mean_py_cov_vi_users, esbm_mean_gn_cov_vi_users]

output_table['VI_items'] = [dc_mean_dp_vi_items, dc_mean_py_vi_items, dc_mean_gn_vi_items, dc_mean_dp_cov_vi_items, dc_mean_py_cov_vi_items, dc_mean_gn_cov_vi_items,
                           esbm_mean_dp_vi_items, esbm_mean_py_vi_items, esbm_mean_gn_vi_items, esbm_mean_dp_cov_vi_items, esbm_mean_py_cov_vi_items, esbm_mean_gn_cov_vi_items]


output_table.to_csv('results/tables/results_simulations.csv', index=False)


model_dp_dc = models_list_out[0]
model_py_dc = models_list_out[1]
model_gn_dc = models_list_out[2]
model_dp_cov_dc = models_list_out[3]
model_py_cov_dc = models_list_out[4]
model_gn_cov_dc = models_list_out[5]
model_dp_esbm = models_list_out[6]
model_py_esbm = models_list_out[7]
model_gn_esbm = models_list_out[8]
model_dp_cov_esbm = models_list_out[9]
model_py_cov_esbm = models_list_out[10]
model_gn_cov_esbm = models_list_out[11]

llk_dp_dc = model_dp_dc.train_llk
llk_py_dc = model_py_dc.train_llk
llk_gn_dc = model_gn_dc.train_llk
llk_dp_dc_cov = model_dp_cov_dc.train_llk
llk_py_dc_cov = model_py_cov_dc.train_llk
llk_gn_dc_cov = model_gn_cov_dc.train_llk
llk_dp_esbm = model_dp_esbm.train_llk
llk_py_esbm = model_py_esbm.train_llk
llk_gn_esbm = model_gn_esbm.train_llk
llk_dp_esbm_cov = model_dp_cov_esbm.train_llk
llk_py_esbm_cov = model_py_cov_esbm.train_llk
llk_gn_esbm_cov = model_gn_cov_esbm.train_llk

plot_heatmap(model_dp_dc, save_path='results/figures/simulations/heatmap_dp_dc_users.png')

groups = [
    {'data': [llk_dp_dc, llk_py_dc, llk_gn_dc], 
     'labels': ['DP', 'PY', 'GN'],
     'title': 'Log-likelihood for DC models',
     'path': 'results/figures/simulations/llk_dc.png'},
    {'data': [llk_dp_dc_cov, llk_py_dc_cov, llk_gn_dc_cov],
     'labels': ['DP', 'PY', 'GN'],
     'title': 'Log-likelihood for DC models with covariates',
     'path': 'results/figures/simulations/llk_dc_cov.png'},
    {'data': [llk_dp_esbm, llk_py_esbm, llk_gn_esbm],
     'labels': ['DP', 'PY', 'GN'],
     'title': 'Log-likelihood for ESBM models',
     'path': 'results/figures/simulations/llk_esbm.png'},
    {'data': [llk_dp_esbm_cov, llk_py_esbm_cov, llk_gn_esbm_cov],
     'labels': ['DP', 'PY', 'GN'],
     'title': 'Log-likelihood for ESBM models with covariates',
     'path': 'results/figures/simulations/llk_esbm_cov.png'}]

for group in groups:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for data, label in zip(group['data'], group['labels']):
        ax.plot(data[2:], label=label)
    
    ax.legend()
    plt.title(group['title'])
    plt.xlabel('Iterations')
    plt.ylabel('Log-likelihood')
    plt.tight_layout()
    plt.savefig(group['path'], dpi=300, bbox_inches='tight')
