import helpers as h

try:
    X_train, X_val, X_test, y_train, y_val, y_test, p_train, p_val, p_test = h.prepare_train_val_test_sets(filenames=['input/dl_X_wl24_sr32_original.pkl', 'input/dl_y_wl24_sr32_original.pkl', 'input/dl_p_wl24_sr32_original.pkl'])
    X_train, X_val, X_test = h.handle_outliers_and_impute(X_train, X_val, X_test, num_mad=4, verbose=True)
    X_train, X_val, X_test = h.scale_features(X_train, X_val, X_test, p_train, p_val, p_test, normalise=False)
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("No errors occurred. Data partitioned successfully.")

#--------------------- Hyperparameter optimisation ---------------------------# 
import numpy as np

from LSTM_VAE import LSTM_VAE
from tensorflow import keras
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping

from hyperopt import hp
from hyperopt import fmin, tpe, Trials

import pickle
import random

from datetime import datetime, timedelta

now = datetime.utcnow() + timedelta(hours=1) # Get current date and time in GMT+1
formatted_now = now.strftime('%y%m%d_%H:%M') # Format as YYMMDD hh:mm
filename = f"{formatted_now}_hyperopt_best_params" # E.g., 240505_09:52_hyperopt_best_params

seed = random.randint(0, 10000)

try:
    # Define the hyperparameter space
    space = {
        'batch_size': hp.choice('batch_size', [32, 64]),
        'int_dim': hp.choice('int_dim', [25, 50, 75, 100, 125, 150, 175, 200]),
        'latent_dim': hp.choice('latent_dim', [7, 8, 10, 12, 14, 18, 24, 32, 48, 72, 96, 120]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.1)),
        'reconstruction_wt': hp.choice('reconstruction_wt', [1, 2, 3]),
        'optimizer': hp.choice('optimizer', ['Adam', 'RMSprop'])
    }

    # Objective function
    def objective(params):
        vae = LSTM_VAE(lstm_input_shape=X_train.shape, int_dim=int(params['int_dim']), latent_dim=int(params['latent_dim']), reconstruction_wt = int(params['reconstruction_wt']), seed=seed)    

        if params['optimizer'] == 'Adam':
            opt = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        elif params['optimizer'] == 'RMSprop':
            opt = keras.optimizers.RMSprop(learning_rate=params['learning_rate'])

        vae.compile(optimizer=opt)

        early_stopping = EarlyStopping(monitor='val_loss', mode="min", patience=10)
        history = vae.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), batch_size=int(params['batch_size']), callbacks=[early_stopping], epochs=1000, verbose=2) # callbacks=[early_stopping, checkpoint]
        return np.min(history.history['val_loss'])

    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=Trials())
    best_params['seed'] = seed
    
    with open(f"{filename}.pkl", 'wb') as f:
        pickle.dump(best_params, f)
except Exception as e:
    print(f"An error occurred: {e}")
else: 
    print(f"No errors occurred. Hyperparameter optimisation procedure completed successfully. Best parameters stored as: {filename}.pkl")