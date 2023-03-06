#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import numpy as np

#Dom Rowan 2023

desc="""
Utility plotting functions
"""

colors = ["#3696ff", "#f70065", "#011a7c", "#761954", "#8800b2"]

def plotparams(ax, labelsize=15):
    '''
    Basic plot params

    :param ax: axes to modify

    :type ax: matplotlib axes object

    :returns: modified matplotlib axes object
    '''
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=labelsize)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax

def plotparams_cbar(cbar):

    cbar.ax.tick_params(direction='out', which='both', labelsize=15)
    cbar.ax.tick_params('y', length=8, width=1.8, which='major')
    cbar.ax.tick_params('y', length=4, width=1, which='minor')

    for axis in ['top', 'bottom', 'left', 'right']:
        cbar.ax.spines[axis].set_linewidth(1.5)

    return cbar

def plt_return(created_fig, fig, ax, savefig, dpi=300):
    if created_fig:
        if savefig is not None:
            fig.savefig(savefig, dpi=dpi)
            return 0
        else:
            plt.show()
            return 0
    else:
        return ax

def fig_init(ax=None, use_plotparams=True, figsize=(12,6),**kwargs):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
        created_fig=True
    else:
        created_fig=False
        fig=None

    if use_plotparams and not has_twin(ax):
        ax = plotparams(ax)

    return fig, ax, created_fig

def has_twin(ax):
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return True
    return False

def format_latex_label(label):

    return label.replace('_', ' ')

def get_colors(vals, cmap='plasma'):
    
    if isinstance(plt.get_cmap(cmap), matplotlib.colors.ListedColormap):
        return [ plt.get_cmap(cmap).colors[i]
                 for i in np.linspace(
                        0, int(0.75*len(plt.get_cmap(cmap).colors)),
                        len(vals), dtype=int)]
    elif isinstance(plt.get_cmap(cmap), matplotlib.colors.LinearSegmentedColormap):
        return [ plt.get_cmap(cmap)(np.arange(0, plt.get_cmap(cmap).N))[i]
                 for i in np.linspace(
                        0, int(0.75*plt.get_cmap(cmap).N),
                        len(vals), dtype=int)]
