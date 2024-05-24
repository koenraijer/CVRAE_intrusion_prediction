import pandas as pd
import helpers as h

def load_and_process(filepath):
    df = pd.read_csv(filepath, sep="\t")

    # Select all rows where `blockcode` contains `TNT`
    df = df[(df['blockcode'].str.contains("TNT_N")) | df['blockcode'].str.contains("TNT_T")]


    # Add time column by combining `date` and `currenttime_plus_ms` columns into a datetime object column with milliseconds
    df['time'] = pd.to_datetime(df['date'] + " " + df['currenttime_plus_ms'], format="%Y-%m-%d %H:%M:%S:%f")

    # Remove cols date, currenttime, currenttime_plus_ms
    df = df.drop(columns=['date', 'currenttime', 'currenttime_plus_ms'])
    df = df[['time', 'blockcode', 'trialcode', 'response', 'blocknum']]

    # Group df by trialnum
    grouped = df.groupby('blocknum')
    # Loop over each groups
    for name, group in grouped:
        response = group[group['trialcode'] == "Intrusion"].iloc[0]['response']
        df.loc[(df['blocknum'] == name) & (df['trialcode'].str.contains("TNT_")), 'response'] = response

    # Remove all rows where trialcode does not contain "TNT_NoThink"
    df = df[df['trialcode'].str.contains("TNT_")]
    df = df.drop(columns=['blocknum', 'blockcode'])

    return df 

def load_inquisit(data_folder = 'input/inquisit', save = False):
    dir_list = h.get_dir_list(data_folder)
    # Filter dir_list for the files of the TNT part (with "part1")
    dir_list = [x for x in dir_list if x.find("part1") != -1]
    df = pd.DataFrame()

    for file in dir_list:
        temp = load_and_process(f"{data_folder}/{file}")
        if not temp.empty:
            df = pd.concat([df, temp])

    if save:
        df.to_csv('output/inquisit_combined_raw.csv', index=False)
    
    return df