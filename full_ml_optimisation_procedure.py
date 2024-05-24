import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Load pickle file')

# Add the arguments
parser.add_argument('PicklePath', metavar='path', type=str, help='the path to pickle file')

# Execute the parse_args() method
args = parser.parse_args()

#--------------------- Loading the data ---------------------------# 
# %pip install tensorflow pandas numpy matplotlib seaborn numpy scikit-learn hyperopt tensorflow_addons pydot graphviz visualkeras reload imblearn neurokit2

import helpers as h
from importlib import reload
from LSTM_VAE import LSTM_VAE
import numpy as np
from tensorflow import keras
import tensorflow as tf

try:
    X_train, X_val, X_test, y_train, y_val, y_test, p_train, p_val, p_test = h.prepare_train_val_test_sets(filenames=['input/dl_X_wl24_sr32_original.pkl', 'input/dl_y_wl24_sr32_original.pkl', 'input/dl_p_wl24_sr32_original.pkl'])
    X_train, X_val, X_test = h.handle_outliers_and_impute(X_train, X_val, X_test, num_mad=4, verbose=True)
    X_train, X_val, X_test = h.scale_features(X_train, X_val, X_test, p_train, p_val, p_test, normalise=False)

    # Concatenate X_train and X_val to create a new X_train, as well as p_train and p_val to create a new p_train, and y_train and y_val to create a new y_train
    X_train_raw = np.concatenate((X_train, X_val), axis=0)
    p_train = np.concatenate((p_train, p_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("No errors occurred. Data partitioned successfully.")

#--------------------- VAE * VAE * VAE ---------------------------#

#--------------------- Training and storing the model ---------------------------# 
import os # https://discuss.tensorflow.org/t/valueerror-when-saving-autoencoder-tf-example/18618
import pickle
from keras.callbacks import EarlyStopping

# Defining hyperparameters
with open(args.PicklePath, 'rb') as f:
    params = pickle.load(f)

# Define the choices for each parameter
choices = {
    'batch_size': [32, 64],
    'int_dim': [25, 50, 75, 100, 125, 150, 175, 200],
    'latent_dim': [7, 8, 10, 12, 14, 18, 24, 32, 48, 72, 96, 120],
    'reconstruction_wt': [1, 2, 3],
    'optimizer': ['Adam', 'RMSprop']
}

# Convert indices to actual parameter values
for param, choices in choices.items():
    params[param] = choices[params[param]]

print("BEST PARAMETERS:\n", params)

# Initializing and compiling the model
vae = LSTM_VAE(lstm_input_shape=X_train_raw.shape, int_dim=int(params['int_dim']), latent_dim=int(params['latent_dim']), reconstruction_wt = int(params['reconstruction_wt']), seed=42)    
if params['optimizer'] == 'Adam':
    opt = keras.optimizers.Adam(learning_rate=params['learning_rate'])
elif params['optimizer'] == 'RMSprop':
    opt = keras.optimizers.RMSprop(learning_rate=params['learning_rate'])
vae.compile(optimizer=opt, run_eagerly=True)

# Define the checkpoint callback
base_model_name = f"model.bs{params['batch_size']}.id{params['int_dim']}.ld{params['latent_dim']}.lr{params['learning_rate']:.3f}.rw{params['reconstruction_wt']}"

# Defining Early stopping
early_stopping = EarlyStopping(monitor='loss', mode="min", patience=5, restore_best_weights = True)

# Fitting the model
vae.fit(x=X_train_raw, y=y_train, batch_size=int(params['batch_size']), callbacks=[early_stopping], epochs=1000, verbose=2)

# --------------------- MACHINE LEARNING * MACHINE LEARNING * MACHINE LEARNING ---------------------------#
# IMPORTS
# import sys
# sys.path.insert(0, '../Analysis')
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload

# ML IMPORTS
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials

# GLOBAL SETTINGS
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:.2f}'.format
plt.rcParams["figure.figsize"] = (20, 10)
plt.style.use('seaborn-v0_8-notebook') # plt.style.use('ggplot'); print(plt.style.available)
pd.set_option('display.max_columns', None)

sr = 32
wl = 24 # Window length in seconds

# Initialise dicts
space_dict = {} # Store the search space for each classifier
model_dict = {
    'xgb': XGBClassifier,
    'glm': LogisticRegression,
    'rf': RandomForestClassifier,
    'svm': SVC
}

# ------------ Defining Hyperparameter Spaces ------------
# ------------ XGBoost ------------

# Define the hyperparameter space
counts = np.unique(y_train, return_counts=True)[1]
scale_pos_weight = counts[0] / counts[1] # Recommended by: https://webcache.googleusercontent.com/search?q=cache:https://towardsdatascience.com/a-guide-to-xgboost-hyperparameters-87980c7f44a9&sca_esv=254eb9c569a53dbc&strip=1&vwsrc=0
# Default recommendations: https://bradleyboehmke.github.io/xgboost_databricks_tuning/tutorial_docs/xgboost_hyperopt.html

xgb_space = {
    'fraction_synthetic': hp.choice('fraction_synthetic', [0, 0.1, 0.25, 0.5, 0.75, 1.0]),
    'window_size': hp.choice('window_size', range(8, 24, 2)),
    'objective':'binary:logistic',
    'max_depth': hp.choice('max_depth', np.arange(2, 11, dtype=int)),
    'min_child_weight': hp.uniform('min_child_weight', 0.1, 15),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(1)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
    'colsample_bynode': hp.uniform('colsample_bynode', 0.5, 1),
    'n_estimators': hp.choice('n_estimators', range(50, 5000)),
    'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_log', np.log(1), np.log(1000))]),
    'reg_lambda': hp.choice('reg_lambda', [0, hp.loguniform('reg_lambda_log', np.log(1), np.log(1000))]),
    'reg_alpha': hp.choice('reg_alpha', [0, hp.loguniform('reg_alpha_log', np.log(1), np.log(1000))]),
    'scale_pos_weight': scale_pos_weight
}

space_dict['xgb'] = xgb_space

# ------------ GLM ------------

# Define the hyperparameter space
glm_space = {
    'fraction_synthetic': hp.choice('fraction_synthetic', [0, 0.1, 0.25, 0.5, 0.75, 1.0]),
    'window_size': hp.choice('window_size', range(8, 24, 2)),
    'C': hp.loguniform('C', np.log(0.001), np.log(1000)),
    'penalty': hp.choice('penalty', ['l1', 'l2']),
    'solver': hp.choice('solver', ['liblinear', 'saga']), # Only solvers that support both L1 and L2 penalties
    'class_weight' : 'balanced',
    'max_iter': 10000
}

space_dict['glm'] = glm_space

# ------------ Random Forest ------------

# Source: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
rf_space = {
    'fraction_synthetic': hp.choice('fraction_synthetic', [0, 0.1, 0.25, 0.5, 0.75, 1.0]),
    'window_size': hp.choice('window_size', range(8, 24, 2)),
    'n_estimators': hp.choice('n_estimators', range(2, 200)),
    'max_depth': hp.choice('max_depth', np.arange(2, 101, dtype=int)),
    'max_features': hp.choice('max_features', ['log2', 'sqrt', None]),
    'min_samples_split': hp.choice('min_samples_split', np.arange(2, 10, dtype=int)),
    'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 5, dtype=int)),
    'class_weight' : 'balanced'
}

space_dict['rf'] = rf_space

# ------------ SVM ------------

# Define the hyperparameter space
svm_space = hp.choice('model_type', [
    {
        'fraction_synthetic': hp.choice('fraction_synthetic_lin', [0, 0.1, 0.25, 0.5, 0.75, 1.0]),
        'window_size': hp.choice('window_size_linear', range(8, 24, 2)),
        'C': hp.loguniform('C_linear', np.log(0.01), np.log(10)),  # Lower range for C
        'kernel': 'linear',
        'class_weight' : 'balanced'
    },
    {
        'fraction_synthetic': hp.choice('fraction_synthetic_rbf', [0, 0.1, 0.25, 0.5, 0.75, 1.0]),
        'window_size': hp.choice('window_size_rbf', range(8, 24, 2)),
        'C': hp.loguniform('C_rbf', np.log(0.01), np.log(10)),
        'kernel': 'rbf',
        'gamma': hp.loguniform('gamma_rbf', np.log(0.001), np.log(1)),  # Lower range for gamma
        'class_weight' : 'balanced'
    }
])

space_dict['svm'] = svm_space

# ------------ Inject Synthetic Data ----------------
def inject_synthetic_data(X_train_fold, y_train_fold, shape, window_size, variational_autoencoder=None, fraction_synthetic=0.5, seed=42, rebalance=True):
    np.random.seed(seed)

    # Calculate the number of synthetic samples to generate
    num_synthetic_samples = int(len(X_train_fold) * fraction_synthetic)
    new_total = len(X_train_fold) + num_synthetic_samples

    synthetic_samples_raw = np.empty((0, *shape[1:]))
    synthetic_labels = np.array([])

    # If rebalance is True, rebalance the class distribution towards the minority class
    if rebalance and num_synthetic_samples > 0:
        # Calculate the number of samples in each class
        num_neg = np.sum(y_train_fold == 0)
        num_pos = np.sum(y_train_fold == 1)

        add_to_neg = max((new_total // 2) - num_neg, 0)
        add_to_pos = max((new_total // 2) - num_pos, 0)

        # Generate synthetic samples for each class
        if add_to_neg > 0:
            synthetic_samples_neg = variational_autoencoder.generate_samples(add_to_neg, condition=0)
            synthetic_samples_raw = np.concatenate([synthetic_samples_raw, synthetic_samples_neg])
            synthetic_labels = np.concatenate([synthetic_labels, np.array([0] * add_to_neg)])

        if add_to_pos > 0:
            synthetic_samples_pos = variational_autoencoder.generate_samples(add_to_pos, condition=1)
            synthetic_samples_raw = np.concatenate([synthetic_samples_raw, synthetic_samples_pos])
            synthetic_labels = np.concatenate([synthetic_labels, np.array([1] * add_to_pos)])

    elif num_synthetic_samples > 1:
        # Generate synthetic samples for each class
        synthetic_samples_neg = variational_autoencoder.generate_samples(num_synthetic_samples // 2, condition=0)
        synthetic_samples_pos = variational_autoencoder.generate_samples(num_synthetic_samples // 2, condition=1)

        # Combine the synthetic samples
        synthetic_samples_raw = np.concatenate([synthetic_samples_neg, synthetic_samples_pos])
        synthetic_labels = np.array([0] * len(synthetic_samples_neg) + [1] * len(synthetic_samples_pos))

    if len(synthetic_samples_raw) > 0:
        synthetic_samples = h.prepare_for_ml(X=synthetic_samples_raw, y=synthetic_labels, wl=window_size)

        # Inject the synthetic samples into the training fold
        X_train_fold = np.concatenate([X_train_fold, synthetic_samples])
        y_train_fold = np.concatenate([y_train_fold, synthetic_labels])

        # Shuffle the training fold
        X_train_fold, y_train_fold = shuffle(X_train_fold, y_train_fold, random_state=seed)

    return X_train_fold, y_train_fold
    
# ------------ Bayesian Hyperparameter Optimisation ------------
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import shuffle
import pickle
import warnings
from datetime import datetime

# Get the current date and time
now = datetime.now()
date_time = now.strftime("%y%m%d_%H%M")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

def optimise_model(model, space, max_evals=100):
    def objective(params):
        # Prepare data
        window_size = params.pop('window_size')  
        fraction_synthetic = params.pop('fraction_synthetic')

        X_train = h.prepare_for_ml(X=X_train_raw, y=y_train, wl=window_size) # p_train, y_train are already defined
        
        # Create folds
        folds = h.create_folds(X_train, y_train, groups=p_train, n_folds=10) # Exhaustive would be 12 folds
        # Train and evaluate the model using cross-validation
        clf = model(**params, random_state=42)
        scores = []

        for train_index, test_index in folds:
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
            X_train_fold, y_train_fold = inject_synthetic_data(X_train_fold, y_train_fold, shape=X_train_raw.shape, variational_autoencoder=vae, fraction_synthetic=fraction_synthetic, window_size=window_size, seed=42, rebalance=True)
            clf.fit(X_train_fold, y_train_fold)
            y_pred = clf.predict(X_test_fold)
            score = balanced_accuracy_score(y_test_fold, y_pred)
            scores.append(score)

        return {'loss': -np.mean(scores), 'status': STATUS_OK, 'params': params, 'scores': scores}

    # Perform the optimisation
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_params = space_eval(space, best)
    best_scores = trials.best_trial['result']['scores']
    return best_params, best_scores

def optimise_models(space_dict, model_dict, max_evals=100):
    best_params_dict = {}
    best_scores_dict = {}
    for key, space in space_dict.items():
        best_params, best_scores = optimise_model(model_dict[key], space, max_evals=max_evals)
        print(f"Best parameters for {key}: {best_params}")
        print(f"Best scores for {key}: {best_scores}")
        best_params_dict[key] = best_params
        best_scores_dict[key] = best_scores
    return best_params_dict, best_scores_dict

best_params_dict, best_scores_dict = optimise_models(space_dict, model_dict, max_evals=100)

# Save the best parameters
with open(f"output/ml/{date_time}_ml_best_params_({base_model_name}).pkl", 'wb') as f:
    pickle.dump(best_params_dict, f)
with open(f"output/ml/{date_time}_ml_best_scores_({base_model_name}).pkl", 'wb') as f:
    pickle.dump(best_scores_dict, f)