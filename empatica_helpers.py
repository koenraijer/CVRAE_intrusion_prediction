import os
import shutil
import re
import pandas as pd
import numpy as np
import neurokit2 as nk
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import helpers as h

def move_folders_to_root(root_path, current_path=None):
    """
    This function deletes the .DS_Store file, recursively searches through all the folders and subfolders, and moves
    folders with no subfolders to the root directory. Empty folders are deleted.
    
    Args:
        root_path (str): The root path where the parent folders are located.
        current_path (str, optional): The current path being processed (used for recursion). Defaults to None.
    
    Returns:
        None
    """
    if current_path is None:
        current_path = root_path

        # Check if the root_path already contains only files and no folders
        items_in_root = [os.path.join(root_path, item) for item in os.listdir(root_path)]
        if all(os.path.isfile(item) for item in items_in_root):
            return
        
    # Remove the .DS_Store file if it exists
    ds_store_path = os.path.join(current_path, '.DS_Store')
    if os.path.exists(ds_store_path):
        os.remove(ds_store_path)

    subfolders = [os.path.join(current_path, item) for item in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, item))]
    
    if not subfolders:
        # Base case: no subfolders
        if current_path != root_path:
            # Move the folder to the root directory
            shutil.move(current_path, get_unique_path(os.path.join(root_path, os.path.basename(current_path))))
    else:
        # Recursively process subfolders
        for subfolder in subfolders:
            move_folders_to_root(root_path, subfolder)

        # Remove empty folders
        if not os.listdir(current_path) and current_path != root_path:
            os.rmdir(current_path)

def get_unique_path(path):
    """
    This function generates a unique path if the given path already exists.
    
    :param path: The path to check for uniqueness.
    :return: A unique path.
    """

    counter = 1
    unique_path = path
    while os.path.exists(unique_path):
        unique_path = f"{path}_{counter}"
        counter += 1
    return unique_path

def clean_folder_name(folder_name):
    """
    Cleans the folder name by removing the trailing '_<number>' or '-<number>'.
    If no trailing number is found, the function returns False.

    Args:
    folder_name (str): The name of the folder to be cleaned.

    Returns:
    str or bool: The cleaned folder name, or False if no trailing number is found.
    """
    # Remove the trailing '_<number>' or '-<number>'
    cleaned_name = re.sub(r'[_-](\d+)$', '', folder_name)
    
    # Check if a trailing number was found and removed
    if cleaned_name == folder_name:
        return False
    
    return cleaned_name

def rename_folders(parent_folder):
    """
    Renames all folders within the specified parent folder.

    Args:
        parent_folder (str): The path to the parent folder.

    Returns:
        None
    """
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)

        # Check if it's a folder
        if os.path.isdir(item_path):
            original_name = item

            cleaned_name = original_name
            previous_cleaned_name = original_name

            while cleaned_name:
                # Clean folder name
                cleaned_name = clean_folder_name(cleaned_name)

                # If there is nothing to clean, break
                if cleaned_name is False:
                    cleaned_name = previous_cleaned_name
                    break

                # Find clean folder name path
                cleaned_path = os.path.join(parent_folder, cleaned_name)

                # If cleaned_path is not unique, set cleaned_name to previous_cleaned_name and break
                if os.path.exists(cleaned_path):
                    cleaned_name = previous_cleaned_name
                    break
                else:
                    previous_cleaned_name = cleaned_name

            cleaned_path = os.path.join(parent_folder, cleaned_name)
            # Rename the folder
            print(f"{original_name} -> {cleaned_name}")
            os.rename(item_path, cleaned_path)

def load_csv_file(file_path):
    df = pd.read_csv(file_path, header=None)
    starting_time = df.iloc[0, 0]
    sampling_rate = int(df.iloc[1, 0])
    signal = df.iloc[2:, :].values
    return starting_time, sampling_rate, signal

def handle_ibi_file(file_path, desired_sampling_rate):
    try:
        df = pd.read_csv(file_path, header=None)
    except pd.errors.EmptyDataError:
        print(f"Skipping {file_path} because it is empty.")
        return pd.DataFrame(), None, None

    if df.shape[0] < 2:
        print(f"Skipping {file_path} due to insufficient signal length.")
        return pd.DataFrame(), None, None
    
    ibi_df = pd.DataFrame(columns=['ibi', 'delta_t', 'datetime', 'unix_time', 'source'])
    ibi_df['ibi'] = df.iloc[1:, 1]
    ibi_df['delta_t'] = df.iloc[1:, 0]

    starting_time = df.iloc[0, 0].astype(int)
    ending_time = starting_time + ibi_df['delta_t'].sum()

    starting_datetime = datetime.fromtimestamp(starting_time)
    ibi_df['datetime'] = [starting_datetime + timedelta(seconds=delta_t) for delta_t in ibi_df['delta_t']]
    ibi_df['unix_time'] = starting_time + ibi_df['delta_t']

    # Convert IBI data into a continuous signal
    ibi_signal = pd.Series(index=ibi_df['unix_time'], data=ibi_df['ibi']).sort_index()
    ibi_signal.index = pd.to_datetime(ibi_signal.index, unit='s')

    # Resample the IBI signal to the desired sampling rate
    ibi_signal = nk.signal_resample(ibi_signal, sampling_rate=1, desired_sampling_rate=desired_sampling_rate, method="FFT")

    ibi_df = pd.DataFrame(ibi_signal, columns=['ibi'])
    ibi_df['datetime'] = ibi_df.index
    ibi_df['unix_time'] = ibi_df.index.astype(int) / 1e9  # Convert datetime index to unix time

    ibi_df.drop(columns='delta_t', inplace=True, errors='ignore')
    file = file_path.split('/')[-2]
    ibi_df['source'] = file

    return ibi_df, starting_time, ending_time

def handle_other_files(file_path, desired_sampling_rate):
    starting_time, sampling_rate, signal = load_csv_file(file_path)
    if signal.shape[0] < 3:
        print(f"Skipping {file_path} due to insufficient signal length.")
        return None, None, None
    signal_data = nk.signal_resample(signal, sampling_rate=sampling_rate, desired_sampling_rate=desired_sampling_rate, method="FFT")
    signal_data = pd.DataFrame(signal_data)
    ending_time = starting_time + signal_data.shape[0] / desired_sampling_rate
    return signal_data, starting_time, ending_time

def load_data_and_combine(folder, desired_sampling_rate=64, verbose=False, useIBI = False):
    has_ibi = False
    file_names_except_ibi = ['ACC.csv', 'TEMP.csv', 'EDA.csv', 'BVP.csv', 'HR.csv']
    data_frames = []
    uniqueness_check = pd.DataFrame(columns=['file_path', 'file_name', 'starting_time', 'ending_time'])
    starting_times = []
    ending_times = []

    for file_name in file_names_except_ibi:
        file_path = os.path.join(folder, file_name)
        data, starting_time, ending_time = handle_other_files(file_path, desired_sampling_rate)
        if data is None:
            continue
        if file_name == 'ACC.csv':  # Handle the ACC.csv file with 3 columns
            data.columns = ['acc_x', 'acc_y', 'acc_z']
        else:
            column_name = file_name.lower().replace('.csv', '')
            data.columns = [column_name]

        starting_times.append(starting_time)
        ending_times.append(ending_time)
        data_frames.append(data)
    
    ibi_df, starting_time, ending_time = handle_ibi_file(os.path.join(folder, 'IBI.csv'), desired_sampling_rate=desired_sampling_rate)
    if useIBI is True and not ibi_df.empty:
        starting_times.append(starting_time)
        ending_times.append(ending_time)
        has_ibi = True

    # Trim dataframes
    trimmed_data_frames, trimmed_ibi_df, latest_start_time, earliest_end_time = trim_dataframes(data_frames, ibi_df, starting_times, ending_times, sr=desired_sampling_rate, has_ibi=has_ibi)
    trimmings = max([df.shape[0] for df in data_frames]) - max([df.shape[0] for df in trimmed_data_frames]) # Number of samples trimmed

    if verbose:
        print(f"latest_start_time: {latest_start_time}, earliest_end_time: {earliest_end_time}.")
        print(f"Original longest dataframe length: {max([df.shape[0] for df in data_frames])}, Trimmed dataframe length: {max([df.shape[0] for df in trimmed_data_frames])}. Difference: {trimmings} samples.")
    
    reset_index_data_frames = [df.reset_index(drop=True) for df in trimmed_data_frames]

    concatenated_df = pd.concat(reset_index_data_frames, axis=1)

    # Add time column starting at latest_start_time and ending at earliest_end_time
    time_unix = np.arange(latest_start_time, earliest_end_time, 1 / desired_sampling_rate)
    datetime = pd.to_datetime(time_unix, unit='s', origin='unix')

    # Add time column to the concatenated_df
    concatenated_df['datetime'] = datetime
    concatenated_df['unix_time'] = time_unix

    # Distill file name from folder (e.g. "file" from "input/empatica/file")
    file = folder.split('/')[-1]
    
    concatenated_df['source'] = file
    ibi_df['source'] = file
    
    return concatenated_df, trimmed_ibi_df, trimmings, uniqueness_check

def trim_dataframes(data_frames, ibi_df, starting_times, ending_times, sr, has_ibi):
    latest_start_time = max(starting_times)
    earliest_end_time = min(ending_times)

    trimmed_data_frames = []
    for idx, df in enumerate(data_frames):
        start_idx = int((latest_start_time - starting_times[idx]) * sr)
        end_idx = int((earliest_end_time - starting_times[idx]) * sr)
        df_trimmed = df.iloc[start_idx:end_idx, :]
        trimmed_data_frames.append(df_trimmed)

    # Trim ibi_df
    if has_ibi:
        trimmed_ibi_df = ibi_df[(ibi_df['unix_time'] >= latest_start_time) & (ibi_df['unix_time'] <= earliest_end_time)]
    else:
        trimmed_ibi_df = None
        
    return trimmed_data_frames, trimmed_ibi_df, latest_start_time, earliest_end_time

def load_empatica(data_folder = 'input/empatica/', useIBI = False, save = False, plotTrimmings=False, desired_sampling_rate=64):
    dir_list = h.get_dir_list(data_folder)
    df = pd.DataFrame() 
    ibi_df = pd.DataFrame()
    trimmings_array = np.array([]) # Trimmings are the numbers of samples removed from dataframes coming from a single folder due to differing start or end times. 
    uniqueness_check_df = pd.DataFrame()
    no_ibi = 0
    for folder in dir_list:
        temp, ibi, trimmings, uniqueness_check = load_data_and_combine(f'input/empatica/{folder}', verbose = False, useIBI = useIBI, desired_sampling_rate=desired_sampling_rate)
        # Concat to df if not empty
        if not temp.empty:
            trimmings_array = np.append(trimmings_array, trimmings)
            df = pd.concat([df, temp])
            uniqueness_check_df = pd.concat([uniqueness_check_df, uniqueness_check])
            if useIBI:
                ibi_df = pd.concat([ibi_df, ibi])
        else:
            if useIBI:
                print(f"Skipping {folder} due to empty dataframe.")
        if ibi is None:
            no_ibi += 1
    
    if plotTrimmings:
        sns.histplot(data=trimmings_array, fill=True)
        plt.title('Number of samples removed from dataframes due to differing start or end times')
        plt.show()
    
    if save:
        if useIBI:
            print(f"Number of folders with no IBI data: {no_ibi} / {len(dir_list)}")
            df.to_csv('output/empatica_raw.csv', index=False)
            ibi_df.to_csv('output/empatica_raw_ibi.csv', index=False)
            print('IBI and Empatica dataframes saved to output folder as: empatica_raw.csv and empatica_raw_ibi.csv.')
        else:
            df.to_csv('output/empatica_raw.csv', index=False)
            print('Empatica dataframe saved to output folder as empatica_raw.csv.')
    return df, ibi_df