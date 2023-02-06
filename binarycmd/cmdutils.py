#!/usr/bin/env python

import numpy as np
import os
import pandas as pd

#Dom Rowan 2023

desc="""
Utility functions for astrocmd
"""

#Check if list tuple, np array, etc
def check_iter(var):
    if isinstance(var, str):
        return False
    elif hasattr(var, '__iter__'):
        return True
    else:
        return False

#utility function for reading in df from various extensions
def pd_read(table_path, low_memory=False):

    if (not isinstance(table_path, pd.core.frame.DataFrame)
        and check_iter(table_path)):
        df0 = pd_read(table_path[0])
        for i in range(1, len(table_path)):
            df0 = pd.concat([df0, pd_read(table_path[i])])
        return df0
    else:
        if type(table_path) == pd.core.frame.DataFrame:
            return table_path
        elif table_path.endswith('.csv') or table_path.endswith('.dat'):
            return pd.read_csv(table_path, low_memory=low_memory)
        elif table_path.endswith('.pickle'):
            return pd.read_pickle(table_path)
        else:
            raise TypeError("invalid extension")

def pd_write(df, table_path, index=False):

    if table_path.endswith('.csv'):
        df.to_csv(table_path, index=index)
    elif table_path.endswith('.pickle'):
        df.to_pickle(table_path, index=index)
    else:
        raise TypeError("invalid extension for ellutils pd write")

#decorator to create a multiprocessing list to store return vals from function 
def manager_list_wrapper(func, L, *args, **kwargs):
    return_vals = func(*args, **kwargs)
    print(return_vals)
    L.append(return_vals)
    return return_vals

def manager_list_wrapper_silent(func, L, *args, **kwargs):
    return_vals = func(*args, **kwargs)
    L.append(return_vals)
    return return_vals


