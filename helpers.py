import os 
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, auc
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import median_abs_deviation

def get_dir_list(path):
    """
    This function returns a list of all the directories in the given path.
    
    :param path: The path to search for directories.
    :return: A list of directories.
    """
    return [x for x in os.listdir(path) if x != ".DS_Store"]

def combine_empatica_and_inquisit(empatica_df, inquisit_df, save=False, sr=64):
    # Add unix_time to df (inquisit_df['time'][0] = 2023-04-18 16:59:11.535)
    inquisit_df['unix_time'] = pd.to_datetime(inquisit_df['time']).astype(int) / 10**9

    # Rename time to datetime
    inquisit_df = inquisit_df.rename(columns={"time": "datetime"})

    # Sort both DataFrames by the 'time' column
    empatica_df = empatica_df.sort_values('unix_time')
    inquisit_df = inquisit_df.sort_values('unix_time')

    merged_df = pd.merge_asof(empatica_df, inquisit_df, on="unix_time", direction="nearest", tolerance=1)
    merged_df['datetime_x'] = pd.to_datetime(merged_df['datetime_x'])
    merged_df['datetime_y'] = pd.to_datetime(merged_df['datetime_y'])
    merged_df['delta_t'] = (merged_df['datetime_x'] - merged_df['datetime_y']).abs()

    # Add a new 'response' column to merged_df containing the response value only if the time difference is less than or equal to 0.25 seconds
    merged_df['new_response'] = merged_df.apply(lambda row: row['response'] if row['delta_t'] <= pd.Timedelta(seconds=1/sr) else 0, axis=1)

    # Add a new 'trialcode' column to merged_df containing the trialcode value only if the time difference is less than or equal to 0.25 seconds, otherwise set it to NaN
    merged_df['trialcode'] = merged_df.apply(lambda row: row['trialcode'] if row['delta_t'] <= pd.Timedelta(seconds=1/sr) else None, axis=1)

    merged_df.drop(columns=['datetime_y', 'response', 'delta_t'], inplace=True)
    merged_df.rename(columns={"datetime_x": "datetime"}, inplace=True)
    merged_df.rename(columns={'new_response': 'response'}, inplace=True)

    # If the response was 3 or 80, set it to 1; if the response was 4 or 81, set it to 2.
    merged_df['response'] = merged_df['response'].apply(lambda x: 1 if x in [3, 80] else 2 if x in [4, 81] else 0)

    # Add a new column named 'intrusion' containing 1 if the response is 1 or 2, and 0 otherwise 
    merged_df['intrusion'] = merged_df.apply(lambda row: 1 if row['response'] in [1, 2] and row['trialcode'] in ["TNT_NoThink_CSm", "TNT_NoThink_CSp"] else 0, axis=1)

    merged_df['intrusion_nothink'] = merged_df.apply(lambda row: 1 if row['response'] in [1, 2] and row['trialcode'] in ["TNT_NoThink_CSm", "TNT_NoThink_CSp"] else 0 if row['trialcode'] in ["TNT_NoThink_CSm", "TNT_NoThink_CSp"] else None, axis=1)
    merged_df['intrusion_tnt'] = merged_df.apply(lambda row: 1 if row['response'] in [1, 2] and row['trialcode'] in ["TNT_NoThink_CSm", "TNT_NoThink_CSp"] else 0 if pd.notna(row['trialcode']) else None, axis=1)

    # Create a dictionary that maps the original filenames to the new filenames
    filename_map = {
        'd1 2': 'pp15_d1 2',
        'd1_1': 'pp13_d1_1',
        '1681713254_A03F6E': 'pp16_1681713254_A03F6E',
        '1681717717_A03F6E': 'pp17_1681717717_A03F6E',
        'd1_3': 'pp18_d1_3',
        'd2_1_1': 'pp17_d2_1_1',
        'd2_2': 'pp16_d2_2',
        'd1': 'pp19_d1',
        'd1_4': 'pp20_d1_4',
        'd2': 'pp18_d2',
        'd2_1': 'pp19_d2_1',
        'd2_4': 'pp20_d2_4'
    }

    # Update the 'source' column
    merged_df['source'] = merged_df['source'].replace(filename_map)

    merged_df['participant'] = merged_df['source'].str.extract('pp(\d{1,2})').astype(float)

    # Rename values for participant to be incremental integers starting from 1
    merged_df['participant'] = merged_df['participant'].replace({3: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16, 20: 17})

    # Turn to int
    merged_df['participant'] = merged_df['participant'].astype(int)

    if save:
        merged_df.to_csv('output/empatica_inquisit_merged.csv', index=False)
    return merged_df

def get_inquisit_tags(filepath):
    """
    This function returns a list of all the timestamps of the inquisit tags in the given file.

    :param filepath: The path to the file to read.
    :return: A list of timestamps.
    """
    df = pd.read_csv(filepath, sep="\t")

    # Select all rows where `blockcode` contains `TNT`
    df = df[(df['blockcode'].str.contains("countdown")) & (df['trialcode'].str.contains("press_button"))]

    # Add time column by combining `date` and `currenttime_plus_ms` columns into a datetime object column with milliseconds
    df['time'] = pd.to_datetime(df['date'] + " " + df['currenttime_plus_ms'], format="%Y-%m-%d %H:%M:%S:%f") - timedelta(hours = 2)

    # Convert 'time' column back to UNIX timestamp
    df['time'] = df['time'].astype('datetime64[ns]').astype(int) / 10**9

    return list(df['time'])

def closest_timestamp(empatica_ts, inquisit_timestamps):
    """
    Find the closest Inquisit timestamp for a given Empatica timestamp.
    
    Args:
        empatica_ts (float): The Empatica timestamp.
        inquisit_timestamps (list): List of Inquisit timestamps.
    
    Returns:
        float: The closest Inquisit timestamp.
    """
    return min(inquisit_timestamps, key=lambda x: abs(x - empatica_ts))

def clean_and_filter(df = None, filename='output/empatica_inquisit_merged.csv', save=False, sr=64, normalise=False, window_length=8):
    print("Preprocessing data...")
    # Data loading
    if df is None:
        df = pd.read_csv(filename)
    
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filtering 
    df['eda'] = nk.eda_clean(df['eda'], sampling_rate=sr, method='biosppy')
    df['bvp'] = nk.ppg_clean(df['bvp'], sampling_rate=sr, heart_rate=None, method='elgendi')

    # Calculate body acceleration
    df['body_acc'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

    # Drop acc columns
    df = df.drop(columns=['acc_x', 'acc_y', 'acc_z'])

    columns_to_normalize = ["body_acc", "temp", "eda", "bvp", "hr"]
    participants = df['participant']

    # Apply StandardScaler to each participant group and avoid resetting the index
    if normalise is True:
        scaled_df = df.groupby(participants)[columns_to_normalize].apply(lambda x: pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns) if len(x) > 1 else x)
        # Reset the inner level of the index
        scaled_df.reset_index(drop=True, inplace=True)

        # Merge the scaled data back into the original dataframe
        df[columns_to_normalize] = scaled_df
        print("Normalised columns: ", columns_to_normalize)
        norm = 'minmax' 
    elif normalise is False:
        scaled_df = df.groupby(participants)[columns_to_normalize].apply(lambda x: pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns) if len(x) > 1 else x)
        # Reset the inner level of the index
        scaled_df.reset_index(drop=True, inplace=True)

        # Merge the scaled data back into the original dataframe
        df[columns_to_normalize] = scaled_df
        print("Normalised columns: ", columns_to_normalize)
        norm = 'standard'
    else:
        norm = 'original'
        pass

    # Create a column 'session_id' that identifies each session
    df['session_id'] = (df['datetime'].diff() > pd.Timedelta(seconds=1/sr)).cumsum()
    # Create a column 'session_duration' that indicates the duration of each session
    df['session_duration'] = df.groupby('session_id')['datetime'].transform(lambda x: x.max() - x.min())
    # Drop all sessions that are shorter than window_length seconds
    print(f"Number of sessions before filtering: {len(df['session_id'].unique())}")
    df = df[df['session_duration'].dt.total_seconds() > window_length]
    print(f"Number of sessions after filtering: {len(df['session_id'].unique())}")

    df = df.drop(columns=['session_duration'])

    # EDA
    signals, info = nk.eda_process(df['eda'], sampling_rate=sr)
    df['eda'] = signals['EDA_Clean'].values
    # df = df.drop(columns=['eda'])

    # df['eda_tonic'] = signals['EDA_Tonic'].values
    # df['eda_phasic'] = signals['EDA_Phasic'].values

    # Print min and max values of each column (df.describe() does not work with the new columns)
    print("Min and max values of each column:")
    print(df[["body_acc", "temp", "eda", "bvp", "hr"]].describe().loc[['min', 'max']])
    # print(df[["body_acc", "temp", "eda_tonic", "eda_phasic", "bvp", "hr"]].describe().loc[['min', 'max']])

    if save:
        df.to_csv(f"output/ei_prep_{norm}.csv", index=False)
    
    print("Preprocessing complete.")
    return df

def prepare_for_vae(sr=32, wl=24, filepath="output/ei_prep.csv", features=['participant', 'temp', 'bvp', 'hr', 'body_acc', 'eda', 'intrusion_nothink'], save=False, data=None, normalise=True, verbose=False):
    if data is not None and not data.empty:
        df = data
    else: 
        df = pd.read_csv(filepath)

    window_length = wl * sr
    window_excess = window_length - (8*sr) if window_length > 8*sr else 0 
    window_length = 8*sr if window_length > 8*sr else window_length

    # df = df[['participant', 'temp', 'bvp', 'hr', 'body_acc', 'eda_tonic', 'eda_phasic', 'intrusion_nothink']]
    df = df[features]

    samples = []
    labels = []
    participants = []

    for i in range(len(df)):
        if df.iloc[i]['intrusion_nothink'] in [0, 1]:
            if i - window_length >= 0 and i + window_excess < len(df):
                if len(df.iloc[i-window_length:i+window_excess]['participant'].unique()) == 1:
                    samples.append(df.iloc[i-window_length:i+window_excess][features[1:-1]].values)
                    labels.append(df.iloc[i]['intrusion_nothink'])
                    participants.append(df.iloc[i]['participant'])

    X = np.array(samples)
    y = np.array(labels)
    p = np.array(participants)

    if verbose:
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"p shape: {p.shape}")
    
    # Apply minmax scaling on a per-participant, per-feature basis
    if normalise is True:
        # Get unique participants
        unique_participants = np.unique(p)

        # Initialize an empty list to hold the scaled data
        scaled_data = []

        # Loop over each unique participant
        for participant in unique_participants:
            # Get the indices of the current participant's data
            participant_indices = np.where(p == participant)[0]

            # Get the current participant's data
            participant_data = X[participant_indices]

            # Apply the scaler to each feature of the participant's data
            for j in range(participant_data.shape[2]):
                # Get the original shape
                original_shape = participant_data[:, :, j].shape

                # Apply the scaler and reshape back to the original shape
                scaled_feature = MinMaxScaler().fit_transform(participant_data[:, :, j].reshape(-1, 1)).reshape(original_shape)

                # Assign the scaled feature back to the participant data
                participant_data[:, :, j] = scaled_feature

            # Append the scaled data to the list
            scaled_data.append(participant_data)

        # Concatenate the list of scaled data arrays along the first axis
        X = np.concatenate(scaled_data, axis=0)
    if normalise is False:
        # Get unique participants
        unique_participants = np.unique(p)

        # Initialize an empty list to hold the scaled data
        scaled_data = []

        # Loop over each unique participant
        for participant in unique_participants:
            # Get the indices of the current participant's data
            participant_indices = np.where(p == participant)[0]

            # Get the current participant's data
            participant_data = X[participant_indices]

            # Apply the scaler to each feature of the participant's data
            for j in range(participant_data.shape[2]):
                # Get the original shape
                original_shape = participant_data[:, :, j].shape

                # Apply the scaler and reshape back to the original shape
                scaled_feature = StandardScaler().fit_transform(participant_data[:, :, j].reshape(-1, 1)).reshape(original_shape)

                # Assign the scaled feature back to the participant data
                participant_data[:, :, j] = scaled_feature

            # Append the scaled data to the list
            scaled_data.append(participant_data)

        # Concatenate the list of scaled data arrays along the first axis
        X = np.concatenate(scaled_data, axis=0)
    else:
        pass
    
    # Obtain eda_tonic and eda_phasic using nk.eda_process. Pay attention to data shape (samples, timesteps, features), where EDA is the last feature (index 4)
    # Reshape X to combine the samples and timesteps dimensions
    X_reshaped = X.reshape(-1, X.shape[-1])

    if verbose:
        print(f"X shape: {X.shape}")
        print(f"X_reshaped shape: {X_reshaped.shape}")

    # Apply eda_process to the reshaped data
    signals, info = nk.eda_process(X_reshaped[:, 4], sampling_rate=sr)
    X_eda_tonic = signals['EDA_Tonic'].values
    X_eda_phasic = signals['EDA_Phasic'].values

    if verbose:
        print(f"X_eda_tonic shape: {X_eda_tonic.shape}")
        print(f"X_eda_phasic shape: {X_eda_phasic.shape}")

    # Reshape the processed data back to the original shape
    X_eda_tonic = X_eda_tonic.reshape(X.shape[0], X.shape[1], -1)
    X_eda_phasic = X_eda_phasic.reshape(X.shape[0], X.shape[1], -1)
    if verbose:
        print(f"X_eda_tonic shape after reshaping: {X_eda_tonic.shape}")
        print(f"X_eda_phasic shape after reshaping: {X_eda_phasic.shape}")

    # Concatenate the processed data with X
    X = np.concatenate((X, X_eda_tonic, X_eda_phasic), axis=-1)
    if verbose:
        print(f"X shape after concatenation: {X.shape}")

    # Remove the EDA feature from X
    X = np.delete(X, 4, axis=-1)
    if verbose:
        print(f"X shape after removing EDA: {X.shape}")

    norm = 'minmax' if normalise is True else ('standard' if normalise is False else 'original')

    if save:
        with open(f'output/dl_X_wl{wl}_sr{sr}_{norm}.pkl', 'wb') as f:
            pickle.dump(X, f)
        with open(f'output/dl_y_wl{wl}_sr{sr}_{norm}.pkl', 'wb') as f:
            pickle.dump(y, f)
        with open(f'output/dl_p_wl{wl}_sr{sr}_{norm}.pkl', 'wb') as f:
            pickle.dump(p, f)
    
    return X, y, p

def prepare_train_val_test_sets(X=None, y=None, p=None, filenames=None, test_size=0.15, val_size=0.05, random_state=42):
    if filenames:
        with open(filenames[0], 'rb') as f:
            X = pickle.load(f)
        with open(filenames[1], 'rb') as f:
            y = pickle.load(f)
        with open(filenames[2], 'rb') as f:
            p = pickle.load(f)

    # Initialize GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Get the indices of the training and test sets
    trainval_idx, test_idx = next(gss.split(X, y, groups=p))

    # Create the training and test sets
    X_train, X_test = X[trainval_idx], X[test_idx]
    y_train, y_test = y[trainval_idx], y[test_idx]

    # Create the training and test groups
    p_train, p_test = p[trainval_idx], p[test_idx]

    # Shuffle the training set
    X_train, y_train, p_train = shuffle(X_train, y_train, p_train, random_state=random_state)

    # Initialize another GroupShuffleSplit
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)

    # Get the indices of the training and validation sets
    train_idx, val_idx = next(gss_val.split(X_train, y_train, groups=p_train))

    # Create the training and validation sets
    X_train, X_val = X_train[train_idx], X_train[val_idx]
    y_train, y_val = y_train[train_idx], y_train[val_idx]

    # Create the training and validation groups
    p_train, p_val = p_train[train_idx], p_train[val_idx]

    # Shuffle the training set
    X_train, y_train, p_train = shuffle(X_train, y_train, p_train, random_state=random_state)

    print("Train size: ", (X_train.shape[0] / X.shape[0]) * 100)
    print("Val size: ", (X_val.shape[0] / X.shape[0]) * 100)
    print("Test size: ", (X_test.shape[0] / X.shape[0]) * 100)

    print("Size: :", X_train.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, p_train, p_val, p_test

def prepare_for_ml(X, y, feature_names=['temp', 'bvp', 'hr', 'body_acc', 'eda_tonic', 'eda_phasic'], wl=24, sr=32):
    timesteps = wl * sr

    X = X[:, :timesteps, :]

    # Define the operations
    operations = [np.mean, np.std, np.min, np.max, skew, kurtosis]
    operation_names = ['mean', 'std', 'min', 'max', 'skew', 'kurt']

    # Initialize an empty list to store the results
    results = []

    # Initialize an empty list to store the column names
    column_names = []

    # Loop over the last dimension of the data (the features)
    for i in range(X.shape[-1]):
        # Extract the feature
        feature = X[:, :, i]

        # Calculate the aggregates for this feature
        aggregates = [op(feature, axis=1) for op in operations]

        # Replace NaN values with the mean of the feature for that decision class
        for j, op in enumerate(operations):
            if np.isnan(aggregates[j]).any():
                for class_value in np.unique(y):
                    mask = (y == class_value)
                    aggregates[j][mask & np.isnan(aggregates[j])] = np.nanmean(aggregates[j][mask])

        # Add the aggregates to the results
        results.extend(aggregates)

        # Add the column names for this feature
        column_names.extend([f'{feature_names[i]}_{op_name}' for op_name in operation_names])

    # Convert the results to a 2D array
    results = np.stack(results, axis=1)

    # Convert the results to a DataFrame
    df = pd.DataFrame(results, columns=column_names)

    return df

def create_folds(X_train, y_train, groups, n_folds=10, verbose=False):
    """
    Create folds for cross-validation using GroupKFold.

    Parameters:
    - X_train (array-like): The input features for training.
    - y_train (array-like): The target variable for training.
    - groups (array-like): The groups to be used for grouping the samples.
    - n_folds (int): The number of folds to create (default=10).

    Returns:
    - folds (list): A list of tuples containing train and test indices for each fold.
    """

    # Create GroupKFold object
    gkf = GroupKFold(n_splits=n_folds)

    # Folds must be a list of tuples of train and test indices
    folds = []
    for train_index, test_index in gkf.split(X_train, y_train, groups):
        folds.append((train_index.tolist(), test_index.tolist()))

    # Print length of each sublist
    if verbose:
        print("Folds created:")
        for fold in folds:
            print(f"Train: {len(fold[0])}, Eval: {len(fold[1])}")

    return folds

def xgb_micro_f1(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(preds)
    return 'micro_f1', -f1_score(labels, preds, average='micro')

def xgb_aucpr(preds, dtrain):
    labels = dtrain.get_label()
    return 'aucpr', -average_precision_score(labels, preds)

def plot_metrics(y_val, y_pred, model, X_val, plot_confusion_matrix=False, plot_auprc=False, plot_auroc=False):
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    if plot_confusion_matrix:
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xticks([0.5, 1.5], ['0', '1'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    # AUPRC curve
    precision, recall, thresholds_auprc = precision_recall_curve(y_val, model.predict_proba(X_val)[:,1])
    auprc_score = auc(recall, precision)
    if plot_auprc:
        plt.plot(recall, precision, marker='.', label='XGBoost')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve: AUC = {:.2f}'.format(auprc_score))
        plt.legend()
        plt.show()

    # AUROC curve
    fpr, tpr, thresholds_auroc = roc_curve(y_val, model.predict_proba(X_val)[:,1])
    auroc_score = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
    if plot_auroc:
        plt.plot(fpr, tpr, marker='.', label='XGBoost')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve: AUC = {:.2f}'.format(auroc_score))
        plt.legend()
        plt.show()

    return {'confusion_matrix': cm, 'auprc': { 'precision': precision, 'recall': recall, 'thresholds': thresholds_auprc, 'auprc_score': auprc_score }, 'auroc': { 'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds_auroc, 'auroc_score': auroc_score }}

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    # We want to show all ticks...
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Define the threshold for color contrast
    threshold = cm.max() / 2.

    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > threshold else "black"
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color=text_color)

    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    plt.show()

def plot_all_metrics(all_model_metrics):
    # Plot AUPRC
    for model_name, metrics in all_model_metrics.items():
        precision = metrics['auprc']['precision']
        recall = metrics['auprc']['recall']
        auprc_score = metrics['auprc']['auprc_score']
        plt.plot(recall, precision, marker='.', label=f'{model_name} AUPRC = {auprc_score:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # Plot AUROC
    for model_name, metrics in all_model_metrics.items():
        fpr = metrics['auroc']['fpr']
        tpr = metrics['auroc']['tpr']
        auroc_score = metrics['auroc']['auroc_score']
        plt.plot(fpr, tpr, marker='.', label=f'{model_name} AUROC = {auroc_score:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Plot confusion matrices
    for model_name, metrics in all_model_metrics.items():
        cm = metrics['confusion_matrix']
        plot_confusion_matrix(cm, f'{model_name} Confusion Matrix')

def get_dataset_names(s_range, reference_classes):
    # function that returns a list of dataset names of format 'dataset_{reference}_win{s}' for a given range of s (including step size) and a list of reference_classes
    dataset_names = []
    for s in s_range:
        for ref in reference_classes:
            dataset_names.append(f'dataset_{ref}_win{s}.csv')
    return dataset_names

def get_dataset_name(s, reference_class):
    return f'dataset_{reference_class}_win{s}.csv'


def handle_outliers_and_impute(X_train, X_val, X_test, random_state=42, num_mad=3, verbose=False):
    # Number of features
    num_features = X_train.shape[2]

    # Impute missing values before outlier detection
    imputer = IterativeImputer(random_state=random_state)

    # Reshape the data to 2D, impute, then reshape back to 3D
    X_train_shape = X_train.shape
    X_val_shape = X_val.shape
    X_test_shape = X_test.shape
    
    X_train = imputer.fit_transform(X_train.reshape(-1, X_train_shape[-1])).reshape(X_train_shape)
    X_val = imputer.transform(X_val.reshape(-1, X_val_shape[-1])).reshape(X_val_shape)
    X_test = imputer.transform(X_test.reshape(-1, X_test_shape[-1])).reshape(X_test_shape)

    print("Initial imputation complete.")

    # Print missing values
    if verbose:
        print("Missing values before outlier detection:")
        print(pd.DataFrame({
            'Train': [np.mean(np.isnan(X_train))],
            'Validation': [np.mean(np.isnan(X_val))],
            'Test': [np.mean(np.isnan(X_test))]
        }))

    # Initialize arrays to store outliers
    outliers_train = np.zeros_like(X_train, dtype=bool)
    outliers_val = np.zeros_like(X_val, dtype=bool)
    outliers_test = np.zeros_like(X_test, dtype=bool)

    # Initialize DataFrame to store percentage of outliers
    outliers_df = pd.DataFrame(columns=['Feature', 'Train', 'Validation', 'Test'])

    for feature in range(num_features):
        # Select the feature from each dataset
        X_train_feature = X_train[:, :, feature]
        X_val_feature = X_val[:, :, feature]
        X_test_feature = X_test[:, :, feature]

        # Median Absolute Deviation
        mad = median_abs_deviation(X_train_feature)
        threshold = num_mad * mad  # 3x median absolute deviation as threshold

        outliers_train[:, :, feature] = np.abs(X_train_feature - np.median(X_train_feature)) > threshold
        outliers_val[:, :, feature] = np.abs(X_val_feature - np.median(X_val_feature)) > threshold
        outliers_test[:, :, feature] = np.abs(X_test_feature - np.median(X_test_feature)) > threshold

        # Add percentage of outliers to DataFrame
        outliers_df = pd.concat([outliers_df, pd.DataFrame({
            'Feature': feature,
            'Train': np.mean(outliers_train[:, :, feature]) * 100,
            'Validation': np.mean(outliers_val[:, :, feature]) * 100,
            'Test': np.mean(outliers_test[:, :, feature]) * 100
        }, index=[0])], ignore_index=True)

    # Replace outliers with np.nan in the original datasets
    X_train = np.where(outliers_train, np.nan, X_train)
    X_val = np.where(outliers_val, np.nan, X_val)
    X_test = np.where(outliers_test, np.nan, X_test)

    # Impute missing values after outlier detection
    X_train = imputer.fit_transform(X_train.reshape(-1, X_train_shape[-1])).reshape(X_train_shape)
    X_val = imputer.transform(X_val.reshape(-1, X_val_shape[-1])).reshape(X_val_shape)
    X_test = imputer.transform(X_test.reshape(-1, X_test_shape[-1])).reshape(X_test_shape)

    print("Final imputation complete.")

    # Print DataFrame of outliers
    if verbose:
        print(outliers_df)

    return X_train, X_val, X_test

def scale_features(X_train, X_val, X_test, p_train, p_val, p_test, normalise=True):
    datasets = [X_train, X_val, X_test]
    participants = [p_train, p_val, p_test]
    scaled_datasets = []

    for X, p in zip(datasets, participants):
        unique_participants = np.unique(p)
        scaled_data = []

        for participant in unique_participants:
            participant_indices = np.where(p == participant)[0]
            participant_data = X[participant_indices]

            for j in range(participant_data.shape[2]):
                original_shape = participant_data[:, :, j].shape

                if normalise:
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()

                scaled_feature = scaler.fit_transform(participant_data[:, :, j].reshape(-1, 1)).reshape(original_shape)
                participant_data[:, :, j] = scaled_feature

            scaled_data.append(participant_data)

        scaled_X = np.concatenate(scaled_data, axis=0)
        scaled_datasets.append(scaled_X)

    return scaled_datasets

def inject_synthetic_data(X_train_fold, y_train_fold, shape, window_size, variational_autoencoder=None, fraction_synthetic=0.5, seed=42, rebalance=True):
    np.random.seed(seed)
    
    # Calculate the number of synthetic samples to generate
    num_synthetic_samples = int(len(X_train_fold) * fraction_synthetic)
    new_total = len(X_train_fold) + num_synthetic_samples

    synthetic_samples_raw = np.empty((0, *shape[1:]))
    synthetic_labels = np.array([])

    if rebalance and num_synthetic_samples > 0:
        # Calculate the number of samples in each class
        num_neg = np.sum(y_train_fold == 0)
        num_pos = np.sum(y_train_fold == 1)

        # Calculate ideal number of samples in each class
        ideal_num_neg = new_total // 2
        ideal_num_pos = new_total - ideal_num_neg

        if num_pos < num_neg:
            # Add to num_pos until it's equal to num_neg, as long as that number is smaller than the num of synthetic samples
            add_to_pos = num_neg - num_pos if num_neg - num_pos < num_synthetic_samples else num_synthetic_samples
            add_to_neg = 0
        elif num_neg < num_pos: 
            # Add to num_neg until it's equal to num_pos, as long as that number is smaller than the num of synthetic samples
            add_to_neg = num_pos - num_neg if num_pos - num_neg < num_synthetic_samples else num_synthetic_samples
            add_to_pos = 0

        # at this point, both classes have been rebalanced, IF there is a remainder. 
        remainder = num_synthetic_samples - add_to_neg - add_to_pos

        # The remainder should be split equally among the two classes
        add_to_pos += remainder // 2
        add_to_neg += remainder - (remainder // 2)

        if add_to_neg > 0:
            synthetic_samples_neg = variational_autoencoder.generate_samples(add_to_neg, condition=0)
            synthetic_samples_raw = np.concatenate([synthetic_samples_raw, synthetic_samples_neg])
            synthetic_labels = np.concatenate([synthetic_labels, np.array([0] * add_to_neg)])
            print(f"Synthetic negative samples generated: {len(synthetic_samples_neg)}")
        
        if add_to_pos > 0:
            synthetic_samples_pos = variational_autoencoder.generate_samples(add_to_pos, condition=1)
            synthetic_samples_raw = np.concatenate([synthetic_samples_raw, synthetic_samples_pos])
            synthetic_labels = np.concatenate([synthetic_labels, np.array([1] * add_to_pos)])
            print(f"Synthetic positive samples generated: {len(synthetic_samples_pos)}")

    elif num_synthetic_samples > 1:
        synthetic_samples_neg = variational_autoencoder.generate_samples(num_synthetic_samples // 2, condition=0)
        synthetic_samples_pos = variational_autoencoder.generate_samples(num_synthetic_samples // 2, condition=1)
        synthetic_samples_raw = np.concatenate([synthetic_samples_neg, synthetic_samples_pos])
        synthetic_labels = np.array([0] * len(synthetic_samples_neg) + [1] * len(synthetic_samples_pos))
        print(f"Synthetic negative samples generated without rebalancing: {len(synthetic_samples_neg)}")
        print(f"Synthetic positive samples generated without rebalancing: {len(synthetic_samples_pos)}")

    if len(synthetic_samples_raw) > 0:
        synthetic_samples = h.prepare_for_ml(X=synthetic_samples_raw, y=synthetic_labels, wl=window_size)
        print(f"Prepared synthetic samples ready for injection: {len(synthetic_samples)}")
        X_train_fold = np.concatenate([X_train_fold, synthetic_samples])
        y_train_fold = np.concatenate([y_train_fold, synthetic_labels])
        print(f"X_train_fold size after injection: {len(X_train_fold)}")
        print(f"y_train_fold size after injection: {len(y_train_fold)}")
        
        X_train_fold, y_train_fold = shuffle(X_train_fold, y_train_fold, random_state=seed)

    return X_train_fold, y_train_fold