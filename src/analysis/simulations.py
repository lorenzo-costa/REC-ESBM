import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aux_functions import plot_heatmap
from esbm_rec import esbm
from dc_esbm_rec import dcesbm
from baseline import Baseline
from nb_functions import compute_log_likelihood
from valid_functs import validate_models, generate_val_set, multiple_runs
import yaml
import pickle

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

mean_dp_mse = np.mean(mse_list[0::6])
mean_py_mse = np.mean(mse_list[1::6])
mean_gn_mse = np.mean(mse_list[2::6])
mean_dp_cov_mse = np.mean(mse_list[3::6])
mean_py_cov_mse = np.mean(mse_list[4::6])
mean_gn_cov_mse = np.mean(mse_list[5::6])

mean_dp_mae = np.mean(mae_list[0::6])
mean_py_mae = np.mean(mae_list[1::6])
mean_gn_mae = np.mean(mae_list[2::6])
mean_dp_cov_mae = np.mean(mae_list[3::6])
mean_py_cov_mae = np.mean(mae_list[4::6])
mean_gn_cov_mae = np.mean(mae_list[5::6])

mean_dp_prec = np.mean(precision_list[0::6])
mean_py_prec = np.mean(precision_list[1::6])
mean_gn_prec = np.mean(precision_list[2::6])
mean_dp_cov_prec = np.mean(precision_list[3::6])
mean_py_cov_prec = np.mean(precision_list[4::6])
mean_gn_cov_prec = np.mean(precision_list[5::6])

mean_dp_rec = np.mean(recall_list[0::6])
mean_py_rec = np.mean(recall_list[1::6])
mean_gn_rec = np.mean(recall_list[2::6])
mean_dp_cov_rec = np.mean(recall_list[3::6])
mean_py_cov_rec = np.mean(recall_list[4::6])
mean_gn_cov_rec = np.mean(recall_list[5::6])

mean_dp_vi_users = np.mean(vi_users_list[0::6])
mean_py_vi_users = np.mean(vi_users_list[1::6])
mean_gn_vi_users = np.mean(vi_users_list[2::6])
mean_dp_cov_vi_users = np.mean(vi_users_list[3::6])
mean_py_cov_vi_users = np.mean(vi_users_list[4::6])
mean_gn_cov_vi_users = np.mean(vi_users_list[5::6])

mean_dp_vi_items = np.mean(vi_items_list[0::6])
mean_py_vi_items = np.mean(vi_items_list[1::6])
mean_gn_vi_items = np.mean(vi_items_list[2::6])
mean_dp_cov_vi_items = np.mean(vi_items_list[3::6])
mean_py_cov_vi_items = np.mean(vi_items_list[4::6])
mean_gn_cov_vi_items = np.mean(vi_items_list[5::6])

output_table = pd.DataFrame()

output_table['Model'] = ['DP', 'PY', 'GN', 'DP_cov', 'PY_cov', 'GN_cov']
output_table['MAE'] = [mean_dp_mae, mean_py_mae, mean_gn_mae, mean_dp_cov_mae, mean_py_cov_mae, mean_gn_cov_mae]
output_table['MSE'] = [mean_dp_mse, mean_py_mse, mean_gn_mse, mean_dp_cov_mse, mean_py_cov_mse, mean_gn_cov_mse]
output_table['Precision'] = [mean_dp_prec, mean_py_prec, mean_gn_prec, mean_dp_cov_prec, mean_py_cov_prec, mean_gn_cov_prec]
output_table['Recall'] = [mean_dp_rec, mean_py_rec, mean_gn_rec, mean_dp_cov_rec, mean_py_cov_rec, mean_gn_cov_rec]
output_table['VI_users'] = [mean_dp_vi_users, mean_py_vi_users, mean_gn_vi_users, mean_dp_cov_vi_users, mean_py_cov_vi_users, mean_gn_cov_vi_users]
output_table['VI_items'] = [mean_dp_vi_items, mean_py_vi_items, mean_gn_vi_items, mean_dp_cov_vi_items, mean_py_cov_vi_items, mean_gn_cov_vi_items]

output_table.to_csv('results/tables/results_simulations.csv', index=False)

model_dp_dc = models_list_out[-6]
model_py_dc = models_list_out[-5]
model_gn_dc = models_list_out[-4]
model_dp_cov_dc = models_list_out[-3]
model_py_cov_dc = models_list_out[-2]
model_gn_cov_dc = models_list_out[-1]
    
llk_dp = model_dp_dc.train_llk
llk_py = model_py_dc.train_llk
llk_gn = model_gn_dc.train_llk

llk_dp_cov = model_dp_cov_dc.train_llk
llk_py_cov = model_py_cov_dc.train_llk
llk_gn_cov = model_gn_cov_dc.train_llk

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(llk_dp[2:], label='DP')
ax[0].plot(llk_py[2:], label='PY')
ax[0].plot(llk_gn[2:], label='GN')
ax[0].legend()

ax[1].plot(llk_dp_cov[2:], label='DP_cov')
ax[1].plot(llk_py_cov[2:], label='PY_cov')
ax[1].plot(llk_gn_cov[2:], label='GN_cov')
ax[1].legend()

plt.title('Log-likelihood for different models')
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')

plt.tight_layout()
plt.savefig('results/figures/simulations/llk_simulations.png')