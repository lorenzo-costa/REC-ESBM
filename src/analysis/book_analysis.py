
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aux_functions import plot_heatmap
from esbm_rec import esbm
from dc_esbm_rec import dcesbm
from baseline import Baseline
from analysis.numba_functions import compute_log_likelihood
from valid_functs import validate_models, generate_val_set, multiple_runs
import seaborn as sns
from expectations import expected_cl_py, HGnedin
import yaml

###########################
# loading stuff
# Load dataset
dataset_clean = pd.read_csv('data/processed/dataset_clean.csv')

# Load settings
with open("src/analysis/config_books.yaml", "r") as f:
    config = yaml.safe_load(f)
    
n_users = config["general_params"]["num_users"]
n_items = config["general_params"]["num_items"]
n_cl_u = config["general_params"]["bar_h_users"]
n_cl_i = config["general_params"]["bar_h_items"]
params_dp = config["params_variants"]["dp"]
params_py = config["params_variants"]["py"]
params_gn = config["params_variants"]["gn"]
params_dp_cov = config["params_variants"]["dp_cov"]
params_py_cov = config["params_variants"]["py_cov"]
params_gn_cov = config["params_variants"]["gn_cov"]
n_iters = config["general_params"]["n_iters"]
burn_in = config["general_params"]["burn_in"]
thinning = config["general_params"]["thinning"]
k = config["general_params"]["k"]
seed = config["general_params"]["seed"]

###########################
# data preparation
###########################
# Create user-item matrix and take subset
matrix_form = dataset_clean.pivot_table(index='user_id', columns='book_id', values='rating', fill_value=0).astype(int)
matrix_form = matrix_form.to_numpy()
matrix_small = matrix_form[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]][:, np.flip(np.argsort(matrix_form.sum(axis=0)))[:n_items]].copy()

fig, ax = plt.subplots(figsize=(10, 6))
plt.title('User-Book Heatmap')
sns.heatmap(matrix_small, ax=ax)
ax.set_xlabel('Books')
ax.set_ylabel('Users')
plt.tight_layout()
plt.savefig('results/figures/books/llk_simulations.png')

# Create user covariates
cov_biography = dataset_clean['biography'].values[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]]
cov_fiction = dataset_clean['fiction'].values[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]]
cov_history = dataset_clean['history'].values[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]]
cov_classic = dataset_clean['classic'].values[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]]
cov_romance = dataset_clean['romance'].values[np.flip(np.argsort(matrix_form.sum(axis=1)))[:n_users]]
cov_items = [
    ('cat_biography', cov_biography[:n_items]),
    ('cat_fiction', cov_fiction[:n_items]),
    ('cat_history', cov_history[:n_items]),
    ('cat_classic', cov_classic[:n_items]),
    ('cat_romance', cov_romance[:n_items])]

# train-test split
Y_train, y_val = generate_val_set(matrix_small, size=0.2, seed=42, only_observed=False)

########################
# training
##########################
# define models
model_list = [esbm, esbm, esbm, esbm, esbm, esbm, dcesbm, dcesbm, dcesbm, dcesbm, dcesbm, dcesbm]
params_list = [params_dp, params_py, params_gn, params_dp_cov, params_py_cov, params_gn_cov, params_dp, params_py, params_gn, params_dp_cov, params_py_cov, params_gn_cov]  
model_names = ['esbm_DP', 'esbm_PY', 'esbm_GN', 'esbm_DP_COV', 'esbm_PY_COV', 'esbm_GN_COV', 'dcesbm_DP', 'dcesbm_PY', 'dcesbm_GN', 'dcesbm_DP_COV', 'dcesbm_PY_COV', 'dcesbm_GN_COV']

# Validate models
out_models = validate_models(Y_train, 
                             y_val, 
                             model_list, 
                             params_list, 
                             n_iters=n_iters, 
                             burn_in=burn_in, 
                             k=k,
                             verbose=1, 
                             thinning=thinning, 
                             model_names=model_names, 
                             print_intermid=True,
                             seed=seed)

#################################
# extract models
##################################
esbm_dp = out_models[0]
esbm_py = out_models[1]
esbm_gn = out_models[2]
esbm_dp_cov = out_models[3]
esbm_py_cov = out_models[4]
esbm_gn_cov = out_models[5]
dcesbm_dp = out_models[6]
dcesbm_py = out_models[7]
dcesbm_gn = out_models[8]
dcesbm_dp_cov = out_models[9]
dcesbm_py_cov = out_models[10]
dcesbm_gn_cov = out_models[11]

llk_esbm_dp = esbm_dp.train_llk
llk_esbm_py = esbm_py.train_llk
llk_esbm_gn = esbm_gn.train_llk
llk_esbm_dp_cov = esbm_dp_cov.train_llk
llk_esbm_py_cov = esbm_py_cov.train_llk
llk_esbm_gn_cov = esbm_gn_cov.train_llk
llk_dcesbm_dp = dcesbm_dp.train_llk
llk_dcesbm_py = dcesbm_py.train_llk
llk_dcesbm_gn = dcesbm_gn.train_llk
llk_dcesbm_dp_cov = dcesbm_dp_cov.train_llk
llk_dcesbm_py_cov = dcesbm_py_cov.train_llk
llk_dcesbm_gn_cov = dcesbm_gn_cov.train_llk

##################################
# plot and save llk plot
###################################
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(llk_esbm_dp, label='esbm_DP')
ax1.plot(llk_esbm_py, label='esbm_PY')
ax1.plot(llk_esbm_gn, label='esbm_GN')
ax1.legend()
plt.title('ESBM Log-Likelihood Convergence')
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.tight_layout()
plt.savefig('results/figures/books/llk_esbm.png')

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(llk_esbm_dp_cov, label='esbm_DP_COV')
ax2.plot(llk_esbm_py_cov, label='esbm_PY_COV')
ax2.plot(llk_esbm_gn_cov, label='esbm_GN_COV')
ax2.legend()
plt.title('ESBM with Covariates Log-Likelihood Convergence')
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.tight_layout()
plt.savefig('results/figures/books/llk_esbm_cov.png')

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(llk_dcesbm_dp, label='dcesbm_DP')
ax3.plot(llk_dcesbm_py, label='dcesbm_PY')
ax3.plot(llk_dcesbm_gn, label='dcesbm_GN')
ax3.legend()
plt.title('DCESBM Log-Likelihood Convergence')
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.tight_layout()
plt.savefig('results/figures/books/llk_dcesbm.png')

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(llk_dcesbm_dp_cov, label='dcesbm_DP_COV')
ax4.plot(llk_dcesbm_py_cov, label='dcesbm_PY_COV')
ax4.plot(llk_dcesbm_gn_cov, label='dcesbm_GN_COV')
ax4.legend()
plt.title('DCESBM with Covariates Log-Likelihood Convergence')
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.tight_layout()
plt.savefig('results/figures/books/llk_dcesbm_cov.png')

######################################
# extract val metrics
########################################
mae_esbm_dp = esbm_dp.mae
mse_esbm_dp = esbm_dp.mse
precision_esbm_dp = esbm_dp.precision
recall_esbm_dp = esbm_dp.recall

mae_esbm_py = esbm_py.mae
mse_esbm_py = esbm_py.mse
precision_esbm_py = esbm_py.precision
recall_esbm_py = esbm_py.recall

mae_esbm_gn = esbm_gn.mae
mse_esbm_gn = esbm_gn.mse
precision_esbm_gn = esbm_gn.precision
recall_esbm_gn = esbm_gn.recall

mae_esbm_dp_cov = esbm_dp_cov.mae
mse_esbm_dp_cov = esbm_dp_cov.mse
precision_esbm_dp_cov = esbm_dp_cov.precision
recall_esbm_dp_cov = esbm_dp_cov.recall

mae_esbm_py_cov = esbm_py_cov.mae
mse_esbm_py_cov = esbm_py_cov.mse
precision_esbm_py_cov = esbm_py_cov.precision
recall_esbm_py_cov = esbm_py_cov.recall

mae_esbm_gn_cov = esbm_gn_cov.mae
mse_esbm_gn_cov = esbm_gn_cov.mse
precision_esbm_gn_cov = esbm_gn_cov.precision
recall_esbm_gn_cov = esbm_gn_cov.recall

mae_dcesbm_dp = dcesbm_dp.mae
mse_dcesbm_dp = dcesbm_dp.mse
precision_dcesbm_dp = dcesbm_dp.precision
recall_dcesbm_dp = dcesbm_dp.recall

mae_dcesbm_py = dcesbm_py.mae
mse_dcesbm_py = dcesbm_py.mse
precision_dcesbm_py = dcesbm_py.precision
recall_dcesbm_py = dcesbm_py.recall

mae_dcesbm_gn = dcesbm_gn.mae
mse_dcesbm_gn = dcesbm_gn.mse
precision_dcesbm_gn = dcesbm_gn.precision
recall_dcesbm_gn = dcesbm_gn.recall

mae_dcesbm_dp_cov = dcesbm_dp_cov.mae
mse_dcesbm_dp_cov = dcesbm_dp_cov.mse
precision_dcesbm_dp_cov = dcesbm_dp_cov.precision
recall_dcesbm_dp_cov = dcesbm_dp_cov.recall

mae_dcesbm_py_cov = dcesbm_py_cov.mae
mse_dcesbm_py_cov = dcesbm_py_cov.mse
precision_dcesbm_py_cov = dcesbm_py_cov.precision
recall_dcesbm_py_cov = dcesbm_py_cov.recall

mae_dcesbm_gn_cov = dcesbm_gn_cov.mae
mse_dcesbm_gn_cov = dcesbm_gn_cov.mse
precision_dcesbm_gn_cov = dcesbm_gn_cov.precision
recall_dcesbm_gn_cov = dcesbm_gn_cov.recall

output_table = pd.DataFrame()
output_table['MAE'] = [mae_esbm_dp, mae_esbm_py, mae_esbm_gn, 
                       mae_esbm_dp_cov, mae_esbm_py_cov, mae_esbm_gn_cov, 
                       mae_dcesbm_dp, mae_dcesbm_py, mae_dcesbm_gn, 
                       mae_dcesbm_dp_cov, mae_dcesbm_py_cov, mae_dcesbm_gn_cov]
output_table['MSE'] = [mse_esbm_dp, mse_esbm_py, mse_esbm_gn, 
                       mse_esbm_dp_cov, mse_esbm_py_cov, mse_esbm_gn_cov,
                       mse_dcesbm_dp, mse_dcesbm_py, mse_dcesbm_gn,
                       mse_dcesbm_dp_cov, mse_dcesbm_py_cov, mse_dcesbm_gn_cov]
output_table['Precision'] = [precision_esbm_dp, precision_esbm_py, precision_esbm_gn,
                             precision_esbm_dp_cov, precision_esbm_py_cov, precision_esbm_gn_cov,
                             precision_dcesbm_dp, precision_dcesbm_py, precision_dcesbm_gn,
                             precision_dcesbm_dp_cov, precision_dcesbm_py_cov, precision_dcesbm_gn_cov]
output_table['Recall'] = [recall_esbm_dp, recall_esbm_py, recall_esbm_gn,
                          recall_esbm_dp_cov, recall_esbm_py_cov, recall_esbm_gn_cov,
                          recall_dcesbm_dp, recall_dcesbm_py, recall_dcesbm_gn,
                          recall_dcesbm_dp_cov, recall_dcesbm_py_cov, recall_dcesbm_gn_cov]

output_table.to_csv('results/tables/results_books.csv', index=False)
