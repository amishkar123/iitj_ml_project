import os
from os.path import join, isfile

import boto3
import pandas as pd

def read_data_from_local_folder(path, verbose=True, ignore_index=False):
    '''
    Read all csv and parquet files with any file name from a folder
    '''
    files_in_folder = [f for f in os.listdir(path) if isfile(join(path, f))]
    data = []
    for file in files_in_folder:
        if verbose:
            print(f'Loading: {path}/{file}')
        if 'csv' in file.rpartition('.')[2]:
            data.append(pd.read_csv(f"{path}/{file}"))
        elif 'parquet' in file.partition('.')[2]:
            data.append(pd.read_parquet(f"{path}/{file}",
                                        engine='pyarrow'))
    return pd.concat(data, ignore_index=ignore_index)