#!/usr/bin/env python

from astropy import log
import matplotlib.pyplot as plt
from matplotib import rc
import numpy as np
import os
import pandas as pd

from . import photsys
from .binary_mist_models import *
from . import plotutils
from . import cmdutils
data_path = os.environ.get('BINARYCMD_DIR', None)

if data_path is None:
    print("Please set the BINARYCMD_DIR environment variable to the path to the data")
    sys.exit(1)

#Dom Rowan 2023

desc="""
Main class for binary star CMD analysis
"""

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

    def mark_id(self, id_list, column='id', plot_kwargs=None):
        
        if plot_kwargs is None:
            plot_kwargs = {}

        if column not in self.df.columns:
            raise ValueError(f'Column {column} not in CMD.df.columns')

        plot_kwargs.setdefault('marker', 'o')
        plot_kwargs.setdefault('color', plotutils.colors[0])
        plot_kwargs.setdefault('edgecolor', 'none')
        plot_kwargs.setdefault('alpha', 0.8)

        if not cmdutils.check_iter(id_list):
            id_list = [ id_list ]

        idx = np.where(self.df[column].isin(id_list))[0]

        if not len(idx):
            log.warning('no targets found to mark')

        for ax in self.ax:
            
            ax.scatter(self.df[self.phot_system.color_corrected].iloc[idx],
                       self.df[self.phot_system.absolute_mag].iloc[idx],
                       **plot_kwargs)

    def plot_hexbin(self, plot_kwargs=None):
        
        if plot_kwargs is None:
            plot_kwargs = {}

        plot_kwargs.setdefault('gridsize', (400, 200))
        plot_kwarga.setdefault('vmin', 10)
        plot_kwargs.setdefault('vmax', 1000)
        plot_kwargs.setdefault('cmap', 'Greys')
        plot_kwargs.setdefault('bins', 'log')

        for ax in self.ax:
            
            ax.hexbin(self.df[self.phot_system.color_corrected],
                      self.df[self.phot_system.absolute_mag],
                      **plot_kwargs)

    def _load_mist_iso(self, fname):

        self.mist_iso = mist_iso(fname, self.phot_system)

    def _load_mist_eep(self, eep_dir, mode='tams', n=1.5):

        self.mist_eep = mist_eep(eep_dir, self.phot_system, iso_result=self.mist_iso,
                                 mode=mode, n=n)

    def plot_mist_iso(self, which='gaia', iso_fname=None, legend=True, simple_color=False):

        if iso_fname is not None:
            self._load_mist_iso(iso_fname, which=which)

        if simple_color:
            colors = ['black', 'black', 'black']
            phase_labels=['_nolegend_']*3
        else:
            colors = ['black', ebutils.colors()[0], ebutils.colors()[2]]
            phase_labels = ['Main Sequence', 'Subgiant Branch', 'Giant Branch']

        for i in range(len(self.mist_iso.ages)):
            if self.mist_iso.ages[i] % 0.5 == 0:
                df = self.mist_iso.df[i]
                if 'iso_phase' in df.columns:
                    for ax in self.ax:
                        ax.plot(df[df.iso_phase == 0].color,
                                df[df.iso_phase == 0][self.phot_system.mist_mag],
                                color=colors[0], lw=2, ls='-', label=phase_labels[0])
                        ax.plot(df[df.iso_phase == 1].color,
                                df[df.iso_phase == 1][self.phot_system.mist_mag],
                                color=colors[1],
                                ls='-',lw=2, label=phase_labels[1])
                        ax.plot(df[df.iso_phase == 2].color,
                                df[df.iso_phase == 2][self.phot_system.mist_mag],
                                color=colors[2],
                                ls='-', lw=2, label=phase_labels[2])

                        phase_labels = ['_nolegend_']*3

        for ax in self.ax:
            sg_xvals = np.linspace(
                    self.mist_iso.sg_spline.x[0],
                    self.mist_iso.color_intersect,
                    100)
            rg_xvals = np.linspace(
                    self.mist_iso.rg_spline.x[0],
                    self.mist_iso.sg_spline(self.mist_iso.color_intersect),
                    100)

            ax.plot(sg_xvals, self.mist_iso.sg_spline(sg_xvals),
                    color=colors[1], ls='--', lw=2)
            ax.plot(self.mist_iso.rg_spline(rg_xvals), rg_xvals,
                    color=colors[2], ls='--', lw=2)

            ax.plot([self.mist_iso.color_intersect,
                     ax.get_xlim()[1]],
                    [self.mist_iso.sg_spline(
                            self.mist_iso.color_intersect)]*2,
                    color=colors[2], ls='--', lw=2)

        if legend:
            self.ax[0].legend(edgecolor='black',
                              fontsize=20, loc='lower left')

    def plot_mist_eep(self, eep_dir=None, iso_fname=None,
                      legend=True, simple_color=False,
                      mode='tams', n=1.5, plot_kwargs=None):

        if iso_fname is not None:
            self._load_mist_iso(iso_fname)

        if eep_dir is not None:
            self._load_mist_eep(eep_dir, mode=mode, n=n)

        if plot_kwargs is None:
            plot_kwargs = {}

        plot_kwargs.setdefault('lw', 2)
        plot_kwargs.setdefault('alpha', 1.0)

        if simple_color:
            colors = ['black', 'black', 'black']
            phase_labels=['_nolegend_']*3
        else:
            colors = ['black', plotutils.colors[0], plotutils.colors[2]]
            phase_labels = ['Main Sequence', 'Subgiant Branch', 'Giant Branch']

        mag0_column = self.phot_system.mist_mag
        mag1_column = self.phot_system.mist_color0
        mag2_column = self.phot_system.mist_color1

        termination_val = self.phot_system.termination_value

        sg_xvals = np.linspace(self.mist_eep.sg_spline.x[0],
                               self.mist_eep.color_intersect, 500)

        if np.abs(termination_val-self.mist_eep.rg_spline(
                self.mist_eep.color_intersect)) > 1e-4:
            #need to identify bp_rp rg termination
            rg_termination = analytic_solvers.binary_search(
                    lambda x: self.mist_eep.rg_spline(x)-termination_val,
                    0.5, 2.5, epsilon=1e-4, plot=False)
            rg_xvals = np.linspace(self.mist_eep.rg_spline.x[0],
                                   rg_termination, 500)
        else:
            rg_xvals = np.linspace(self.mist_eep.rg_spline.x[0],
                                   self.mist_eep.color_intersect, 500)

        for i in range(len(self.mist_iso.ages)):
            if self.mist_iso.ages[i] % 0.5 == 0:
                df = self.mist_iso.df[i]

                idx_change = np.where(
                        (df.phase == 2) &
                        (df[mag0_column] > self.mist_eep.rg_spline(df.color) ))[0]

                phase = df.phase.copy().to_numpy()
                phase[idx_change] = 1
                df['eep_phase'] = phase

                for ax in self.ax:
                    ax.plot(df[df.eep_phase == 0].color,
                            df[df.eep_phase == 0][mag0_column],
                            color=colors[0], ls='-',
                            label=phase_labels[0], **plot_kwargs)
                    ax.plot(df[df.eep_phase == 1].color,
                            df[df.eep_phase == 1][mag0_column],
                            color=colors[1],
                            ls='-', label=phase_labels[1], **plot_kwargs)
                    ax.plot(df[df.eep_phase == 2].color,
                            df[df.eep_phase == 2][mag0_column],
                            color=colors[2],
                            ls='-', label=phase_labels[2], **plot_kwargs)

                    phase_labels = ['_nolegend_']*3

        for ax in self.ax:
            ax.plot(sg_xvals, self.mist_eep.sg_spline(sg_xvals),
                    color=colors[1], ls='--', **plot_kwargs)
            ax.plot(rg_xvals, self.mist_eep.rg_spline(rg_xvals),
                    color=colors[1], ls='--', **plot_kwargs)
            ax.plot([self.mist_eep.color_intersect, ax.get_xlim()[1]],
                    [self.mist_eep.sg_spline(
                            self.mist_eep.color_intersect)]*2,
                    color=colors[2], ls='--', **plot_kwargs)

        if legend:
            self.ax[0].legend(edgecolor='black',
                              fontsize=20, loc='lower left')

    def plot_single_star_isochrone(self, path=None,
                                   plot_kwargs=None):
        
        if path is not None:
            self.mist_iso_path = path

        isocmd = read_mist_models.ISOCMD(self.mist_iso_path)

        age_idx = isocmd.age_index(8.0)
        mag0 = isocmd.isocmds[age_idx][self.phot_system.mist_mag0]
        mag1 = isocmd.isocmds[age_idx][self.phot_system.mist_color0]
        mag2 = isocmd.isocmds[age_idx][self.phot_system.mist_color1]

        phase = isocmd.isocmds[age_idx]['phase']

        df = pd.DataFrame({mag0_column:mag0, 'color':mag1-mag2, 'phase':phase})
        df = df[df.phase == 0].reset_index(drop=True)

        if plot_kwargs is None:
            plot_kwargs = {}

        plot_kwargs.setdefault('color', 'black')
        plot_kwargs.setdefault('lw', 2)
        plot_kwargs.setdefault('ls', '--')
        plot_kwargs.setdefault('label', 'Single Star Isochrone')

        for ax in self.ax:
            ax.plot(df.color, df[mag0_column], **plot_kwargs)

    def _identify_components_iso(self):

        mag0_column = self.phot_system.mist_mag
        mag1_column = self.phot_system.mist_color0
        mag1_column = self.phot_system.mist_color1

        data_absolute = self.phot_system.absolute_mag
        data_color = self.phot_system.color_corrected
        
        idx_sg = np.where(
                ((self.df[data_absolute]) < self.mist_iso.sg_spline(
                        self.df[data_color])) &
                (self.df[data_color] < self.mist_iso.rg_spline(
                        self.df[data_absolute])))[0]

        idx_rg = np.where(
                (self.df[data_absolute] < self.mist_iso.sg_spline(
                        self.mist_iso.color_intersect)) &
                (self.df[data_color] >
                        self.mist_iso.rg_spline(self.df[data_absolute])))[0]

        idx_ms = np.where(
                ((self.df[data_absolute] > self.mist_iso.sg_spline(
                        self.mist_iso.color_intersect)) |
                 (self.df[data_color] < self.mist_iso.rg_spline(
                        self.df[data_aboslute]))))[0]

        return idx_ms, idx_sg, idx_rg

    def _identify_components_eep(self):
        
        mag0_column = self.phot_system.mist_mag
        mag1_column = self.phot_system.mist_color0
        mag1_column = self.phot_system.mist_color1

        data_absolute = self.phot_system.absolute_mag
        data_color = self.phot_system.color_corrected

        idx_sg = np.where(
                (self.df[data_absolute] < self.mist_eep.sg_spline(
                        self.df[data_color])) &
                (self.df[data_absolute] > self.mist_eep.rg_spline(
                        self.df[data_color])) &
                (self.df[data_absolute] < self.mist_eep.sg_spline(
                        self.mist_eep.color_intersect)))[0]

        idx_rg = np.where(
                (self.df[data_absolute] < self.mist_eep.rg_spline(
                        self.df[data_color])) &
                (self.df[data_absolute] < self.mist_eep.sg_spline(
                        self.mist_eep.color_intersect)))[0]

        idx_ms = np.where(
                (self.df[data_absolute] > self.mist_eep.sg_spline(
                        self.df[data_color])) |
                (self.df[data_absolute] > self.mist_eep.sg_spline(
                        self.mist_eep.color_intersect)))[0]

        return idx_ms, idx_sg, idx_rg

    def identify_components(self, method='eep', outtable=None):
        

        if method == 'iso':
            if self.mist_iso is None:
                raise ValueError('no mist iso model defined')
            idx_ms, idx_sg, idx_rg = self._identify_components_iso()
        elif method == 'eep':
            if self.mist_eep is None:
                raise ValueError('no mist eep model defined')
            idx_ms, idx_sg, idx_rg = self._identify_components_eep()
        else:
            raise ValueError(f'invalid method {method}')


        
        idx_none = np.setdiff1d(
                self.df.index.to_numpy(),
                np.concatenate([idx_ms, idx_sg, idx_rg]))

        state = np.zeros(len(self.df), dtype='object')
        state[idx_ms] = 'ms'
        state[idx_sg] = 'sg'
        state[idx_rg] = 'rg'
        state[idx_none] = 'other'

        self.df['state'] = state

        if outtable is not None:
            self.df[['id', 'state']].to_csv(outtable, index=False)

        if 'id' in self.df.columns:
            return self.df[['id', 'state']]
        else:
            return self.df