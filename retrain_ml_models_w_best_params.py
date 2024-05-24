#--------------------- Loading the data ---------------------------# 
# %pip install tensorflow pandas numpy matplotlib seaborn numpy scikit-learn hyperopt tensorflow_addons pydot graphviz visualkeras reload imblearn neurokit2

import helpers as h
import numpy as np

try:
    X_train, X_val, X_test, y_train, y_val, y_test, p_train, p_val, p_test = h.prepare_train_val_test_sets(filenames=['input/dl_X_wl24_sr32_original.pkl', 'input/dl_y_wl24_sr32_original.pkl', 'input/dl_p_wl24_sr32_original.pkl'])
    X_train, X_val, X_test = h.handle_outliers_and_impute(X_train, X_val, X_test, num_mad=4, verbose=True)
    X_train, X_val, X_test_raw = h.scale_features(X_train, X_val, X_test, p_train, p_val, p_test, normalise=False)

    # Concatenate X_train and X_val to create a new X_train, as well as p_train and p_val to create a new p_train, and y_train and y_val to create a new y_train
    X_train_raw = np.concatenate((X_train, X_val), axis=0)
    p_train = np.concatenate((p_train, p_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)

except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("No errors occurred. Data partitioned successfully.")
    # Print statement for lengths of all sets
    # print(f'Length of X_train: {len(X_train)}')
    # print(f'Length of X_val: {len(X_val)}')
    # print(f'Length of X_test: {len(X_test)}')
    # print(f'Length of y_train: {len(y_train)}')
    # print(f'Length of y_val: {len(y_val)}')
    # print(f'Length of y_test: {len(y_test)}')
    # print(f'Length of p_train: {len(p_train)}')
    # print(f'Length of p_val: {len(p_val)}')
    # print(f'Length of p_test: {len(p_test)}')
    # print(f'Length of X_train_raw: {len(X_train_raw)}')

# ----------------- DEEP LEARNING -----------------------
import os # https://discuss.tensorflow.org/t/valueerror-when-saving-autoencoder-tf-example/18618
import pickle
from keras.callbacks import EarlyStopping
from LSTM_VAE import LSTM_VAE
from tensorflow import keras
import tensorflow as tf

# Defining hyperparameters
with open("240507_13:08_hyperopt_best_params.pkl", 'rb') as f:
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
early_stopping = EarlyStopping(monitor='loss', mode="min", patience=10, restore_best_weights = True)

# Fitting the model
vae.fit(x=X_train_raw, y=y_train, batch_size=int(params['batch_size']), callbacks=[early_stopping], epochs=1000, verbose=2)

# ---------------------- MACHINE LEARNING EVALUATION -----------------------------

import pickle
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import warnings
from datetime import datetime

# Get the current date and time
now = datetime.now()
date_time = now.strftime("%y%m%d_%H%M")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# Load the best parameters
with open('output/ml/240518_1558_ml_best_params_(model.bs32.id50.ld18.lr0.002.rw1).pkl', 'rb') as f:
    best_params_dict = pickle.load(f)

# Define the models
models = {
    'xgb': XGBClassifier,
    'glm': LogisticRegression,
    'rf': RandomForestClassifier,
    'svm': SVC
}

def train_models(models, best_params, X_train_raw, y_train, X_test_raw, y_test):
    trained_models = {}
    confusion_matrices = {}
    metrics = {}
    
    for model_name, model_class in models.items():
        params = best_params.get(model_name, {})
        window_size = params.pop('window_size')  
        fraction_synthetic = params.pop('fraction_synthetic')
        print("NEW ITERATION ---------------------------------------")
        print("BEFORE ANY PREPROCESSING: ", len(X_train_raw), len(y_train))
        
        X_train = h.prepare_for_ml(X=X_train_raw, y=y_train, wl=window_size) # p_train, y_train are already defined

        print("AFTER PREPARE FOR ML: ", len(X_train_raw), len(y_train))
        
        X_train_synth, y_train_synth = h.inject_synthetic_data(X_train_fold=X_train, y_train_fold=y_train, shape=X_train_raw.shape, variational_autoencoder=vae, fraction_synthetic=fraction_synthetic, window_size=window_size, seed=42, rebalance=True)

        print("AFTER INJECT SYNTHETIC DATA: ", len(X_train_synth), len(y_train_synth))
        
        if model_name == 'svm':
            model = model_class(**params, probability=True)
        else:
            model = model_class(**params)
        model.fit(X_train_synth, y_train_synth)
        trained_models[model_name] = model
        
        X_test = h.prepare_for_ml(X=X_test_raw, y=y_test, wl=window_size) # p_train, y_train are already defined

        # Evaluate on hold-out set: confusion matrices with TPR and FPR, balanced accuracy, F1-score, AUPRC, AUROC, ROC curve, PR curve
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities needed for AUC and PR

        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
        
        metrics[model_name] = {
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auprc': average_precision_score(y_test, y_proba),
            'auroc': roc_auc_score(y_test, y_proba),
            'roc_curve': roc_curve(y_test, y_proba),
            'pr_curve': precision_recall_curve(y_test, y_proba)
        }

    return trained_models, confusion_matrices, metrics

# Retrain the models with loaded parameters
trained_models, confusion_matrices, metrics = train_models(models, best_params_dict, X_train_raw, y_train, X_test_raw, y_test)

with open(f"output/ml/{date_time}_ml_trained_models.pkl", 'wb') as f:
    pickle.dump(trained_models, f)
with open(f"output/ml/{date_time}_ml_confusion_matrices.pkl", 'wb') as f:
    pickle.dump(confusion_matrices, f)
with open(f"output/ml/{date_time}_ml_evaluation_metrics.pkl", 'wb') as f:
    pickle.dump(metrics, f)