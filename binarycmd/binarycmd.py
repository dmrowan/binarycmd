#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotib import rc
import numpy as np
import os
import pandas as pd

import photsys
import plotutils
import cmdutils
data_path = os.environ.get('BINARYCMD_DIR', None)

if data_path is None:
    print("Please set the BINARYCMD_DIR environment variable to the path to the data")
    sys.exit(1)

class CMD:

    def __init__(self, catalog, ax=None, figsize=(10, 12),
                 mode='tams', n=1.5, 
                 phot_system=photsys.PhotSystem.gaia_default()):

        self.df = cmdutils.pd_read(catalog)
        self.phot_system = phot_system

        self.fig, self.ax, self.created_fig = plotutils.fig_init(
                ax=ax, figsize=figsize, use_plotparams=False)

        if self.created_fig:
            self.fig.subplots_adjust(top=.98, right=.98)

        if not asassnutils.check_iter(self.ax):
            self.ax = [self.ax]

        for a in self.ax: a = plotutils.plotparams(a, labelsize=20)

        self.set_xlim(*self.filter_system.plt_range[0])
        self.set_ylim(*self.filter_system.plt_range[1])

        self.set_xlabel(self.filter_system.xlabel)
        self.set_ylabel(self.filter_system.ylabel)

        self.mist_iso_path = os.path.join(data_path, 'MIST_iso_61ddebc974204.iso.cmd')
        self._load_mist_iso(self.mist_iso_path)

        self.eep_dir = os.path.join(data_path, 'mist_evolutionary_tracks')
        self._load_mist_eep(self.eep_dir, mode=mode, n=n)


    def _check_columns(self):
        
        pass

    def set_xlim(self, left, right):

        for a in self.ax:
            a.set_xlim(left, right)

    def set_ylim(self, bottom, top):

        for a in self.ax:
            a.set_ylim(bottom, top)

            if bottom<top:
                a.invert_yaxis()

    def set_xlabel(self, label):
        for a in self.ax:
            a.set_xlabel(label, fontsize=30)

    def set_ylabel(self, label):
        for a in self.ax:
            a.set_ylabel(label, fontsize=30)


    def calculate_color_mags(self, drop_na=True):
        '''
        calculate the extinction corrected absolute magnitude and color
        '''

        if self.phot_system.color_corrected not in self.df.columns:
            
            self.df[self.phot_system.color_corrected ] = (
                    self.df[self.phot_system.color] - ( self.df[self.phot_system.mwdust_c0] - 
                                                        self.df[self.phot_system.mwdust_c1])

        if self.phot_system.absolute_mag not in self.df.columns:
            
            self.df[self.phot_system.absolute_mag] = (
                    self.df[self.phot_system.mag] - 5*np.log10(self.df.rpgeo)
                    +5-self.df[self.phot_system.mwdust_mag])

        subset = [self.phot_system.absolute_mag, self.phot_system.color_corrected]

        self.df = self.df.replace([-np.inf, np.inf], np.nan)
        if drop_na:
            self.df = self.df.dropna(subset=subset).reset_index(drop=True)

    def apply_quality_cuts(self, avcutoff=2.0, rplx=10):
        
        if 'parallax_over_error' in self.df.columns:
            parallax_over_error_column = 'parallax_over_error'
        elif 'RPlx' in self.df.columns:
            parallax_over_error_column = 'RPlx'
        else:
            raise ValueError("no parallax_over_error column in catalog")


        idx_filter = np.where(
                (self.df[parallax_over_error_column] > rplx) &
                (self.df.mwdust_av < avcutoff))[0]

        self.df = self.df.iloc[idx_filter].reset_index(drop=True)

    def plot(self, color_column=None, cbar_label=None, cmap=cmr.ember_r,
             savefig=None, plot_kwargs=None, cbar_kwargs=None, dpi=200):


        if plot_kwargs is None:
            plot_kwargs = {}

        if cbar_kwargs is None:
            cbar_kwargs = {}

        plot_kwargs.setdefault('marker', '.')
        plot_kwargs.setdefault('edgecolor', 'none')
        plot_kwargs.setdefault('rasterized', True)
        plot_kwargs.setdefault('alpha', 0.4)
        plot_kwargs.setdefault('s', 80)

        if not asassnutils.check_iter(color_column):
            color_column = [color_column]*len(self.ax)

        if not asassnutils.check_iter(cbar_label):
            cbar_label = [cbar_label]*len(self.ax)

        if not asassnutils.check_iter(cmap):
            cmap = [cmap]*len(self.ax)
        elif isinstance(cmap, dict):
            cmap = [cmap]*len(self.ax)

        continuous_color = []
        for i in range(len(color_column)):
            if color_column[i] is None:
                continuous_color.append(None)
            elif isinstance(cmap[i], dict):
                continuous_color.append(False)
            else:
                continuous_color.append(
                        len(self.df[color_column[i]].value_counts().index) > 5)

        assert(len(color_column) == len(cbar_label) == len(cmap))

        for i in range(len(self.ax)):

            if color_column[i] is not None:
                if continuous_color[i]:
                    plot_kwargs['c'] = self.df[color_column[i]].to_numpy()
                    plot_kwargs['cmap'] = cmap[i]
                    plot_kwargs.pop('color', None)
                else:
                    if not isinstance(cmap[i], dict):
                        cmap_dict = dict(zip(
                                self.df[color_column[i]].value_counts().index,
                                plotutils.get_colors(
                                        self.df[color_column[i]].value_counts(),
                                        cmap[i])))
                        cmap[i] = cmap_dict
                    plot_kwargs['c'] = [
                            cmap[i][x] for x in self.df[color_column[i]]]
                    plot_kwargs.pop('cmap', None)
            else:
                plot_kwargs.setdefault('color', 'gray')
                plot_kwargs.pop('c', None)
                plot_kwargs.pop('cmap', None)

            sc = self.ax[i].scatter(self.df[self.phot_system.color_corrected],
                                    self.df[self.phot_system.absolute_mag],
                                    **plot_kwargs)

            if color_column[i] is not None:
                if continuous_color[i]:
                    cax = make_axes_locatable(self.ax[i]).append_axes(
                            'right', size='5%', pad=0.05)
                    cbar = plt.colorbar(sc, cax, orientation='vertical',
                                        **cbar_kwargs)
                    cbar = plotutils.plotparams_cbar(cbar)
                    if cbar_label[i] is None:
                        cbar_label[i] = plotutils.format_latex_label(
                                color_column[i])
                    cbar.set_label(cbar_label[i], fontsize=30)
                    cbar.set_alpha(1)
                    cbar.solids.set(alpha=1)
                else:
                    current_xlim = self.ax[i].get_xlim()
                    current_ylim = self.ax[i].get_ylim()

                    if current_xlim[1] > current_xlim[0]:
                        for k in cmap[i].keys():
                            self.ax[i].scatter(
                                    [current_xlim[1]+1],
                                    [current_ylim[0]],
                                    color=cmap[i][k],
                                    label=plotutils.format_latex_label(k),
                                    **{k:plot_kwargs[k] for k in [
                                            'edgecolor',
                                            'marker',
                                            'alpha']})
                    self.ax[i].set_xlim(current_xlim)
                    self.ax[i].set_ylim(current_ylim)
                    self.ax[i].legend(edgecolor='black', fontsize=15,
                                      markerscale=3)

        if savefig is not None:
            self.fig.savefig(savefig, dpi=dpi)

    def _load_mist_iso(self, fname):

        self.mist_iso = mist_iso(fname, self.phot_system)

    def _load_mist_eep(self, eep_dir, mode='tams', n=1.5, self.phot_system):

        self.mist_eep = 

