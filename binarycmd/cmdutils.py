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


#Binary search method for nonlinear equations
def binary_search(f, x1, x2, epsilon=1e-6, plot=False, ax=None):
    #Check if we can solve for a root in this interval
    if f(x1)*f(x2) > 0:
        return float('NaN')
    xp = .5*(x1+x2) #calculate x prime
    xvals = [xp] #save values as we iterate

    counter = 0
    #Loop until we reach precision (or stumble onto value)
    while (f(xp)!=0) and (abs(x1-x2)>epsilon):
        #Redefine our bracket
        if f(x1)*f(xp) < 0:
            x2 = xp
        else:
            x1 = xp
        xp = .5*(x1+x2)
        xvals.append(xp)

    #Option to plot
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax = plotutils.plotparams(ax)
        ax.plot(xvals, color='xkcd:azure', label='Binary Search')

    #Returned solved value and plot
    if plot and ax is not None:
        return xp, ax
    else:
        return xp
