#!/usr/bin/env python

import argparse
from astropy import log
from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import pandas as pd

from .binarycmd import CMD
from . import cmdutils
from . import plotutils

#Dom Rowan 2023

desc="""
Quick CMD plotter for list of Gaia DR3 sources
"""

def get_data_file(filename):
    return pkg_resources.resource_filename(__name__, '../data/' + filename)

class Star:

    def __init__(self, Source, mwdust_ext=False):

        self.Source = Source

        self.query_gaia()
        self.query_dist()
        if mwdust_ext:
            self.query_mwdust()
        self.calculate_color_mags()

    def query_gaia(self):

        r = Vizier(catalog="I/355/gaiadr3",
                   columns=['Source', 'Gmag', 'BP-RP', 'RPlx',
                            'AG', 'E(BP-RP)',
                            'GLON', 'GLAT']).query_constraints(
                                    Source=str(self.Source))[0]

        r = r.to_pandas().iloc[0].to_dict()

        for k in r.keys():
            if k != 'Source':
                setattr(self, k.replace('-', '_'), r[k])

    def query_dist(self):

        r = Vizier(catalog="I/352/gedr3dis",
                   columns=['Source', 'rpgeo']).query_constraints(
                        Source=str(self.Source))[0]

        r = r.to_pandas().iloc[0].to_dict()

        self.rpgeo = r['rpgeo']

    def query_mwdust(self):

        from . import get_extinctions

        r = get_extinctions.evaluate_map(self.Source, self.GLON, self.GLAT, self.rpgeo)
        self.mwdust_av = r[1]
        self.mwdust_ah = r[2]
        self.mwdust_aj = r[3]
        self.mwdust_ak = r[4]

        self.AG = self.mwdust_av*0.789
        self.abp = self.mwdust_av*1.002
        self.arp = self.mwdust_av*0.589
        self.E_BP_RP_ = self.abp-self.arp

    @property
    def quality_filter(self):
        return (self.RPlx > 10) and (self.mwdust_av < 2)

    def calculate_color_mags(self):

        if np.isnan(self.AG) or np.isnan(self.E_BP_RP_):
            self.absolute_g = self.Gmag - 5*np.log10(self.rpgeo)+5
            self.bp_rp_corrected = self.BP_RP
        else:
            self.absolute_g = self.Gmag - 5*np.log10(self.rpgeo)+5 - self.AG
            self.bp_rp_corrected = self.BP_RP - self.E_BP_RP_


def plot(source_list, ax=None, savefig=None,
         plot_kwargs=None, 
         star_list=None,
         mwdust_ext=False,xlim=None, ylim=None,
         save_output=None,
         background=get_data_file('random_gaia.csv')):

    if not cmdutils.check_iter(source_list):

        source_list = [source_list]

    if star_list is None:
        star_list = []
        for source in source_list:
            try:
                star = Star(source, mwdust_ext=mwdust_ext)
                star_list.append(star)
            except:
                continue
        #star_list = [Star(source, mwdust_ext=mwdust_ext) for source in source_list]
        if len(star_list) == 0:
            print(source_list)

    fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(8, 10))
    if created_fig:
        fig.subplots_adjust(top=.98, right=.98)

    if plot_kwargs is None:
        plot_kwargs = {}

    plot_kwargs.setdefault('color', 'xkcd:red')
    plot_kwargs.setdefault('marker', 'o')
    plot_kwargs.setdefault('s', 150)
    plot_kwargs.setdefault('edgecolor', 'black')
    plot_kwargs.setdefault('alpha', 0.8)

    if xlim is None:
        xlim = (-0.6, 2.4)
    if ylim is None:
        ylim = (-4.1, 8.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()

    xlabel = r'$G_{\rm{BP}}-G_{\rm{RP}}$ (mag)'
    ylabel = r'$M_G$ (mag)'

    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)

    if background is not None:
        df_bkg = cmdutils.pd_read(background)
        ax.scatter(df_bkg.bp_rp_corrected, df_bkg.absolute_g, color='gray', marker='.',
                   edgecolor='none', alpha=0.6, rasterized=True)

    source = [ s.Source for s in star_list ]
    bp_rp = [ s.bp_rp_corrected for s in star_list ]
    absolute_g = [ s.absolute_g for s in star_list ]

    df_out = pd.DataFrame({'Source':source, 'bp_rp':bp_rp, 'mg':absolute_g})
    print(df_out)

    ax.scatter(bp_rp, absolute_g, **plot_kwargs)

    if save_output is not None:
        df_out.to_csv(save_output, index=False)

    return plotutils.plt_return(created_fig, fig, ax, savefig)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--source', nargs='+', help='Gaia DR3 Source', default=None)
    parser.add_argument('--background', default=get_data_file('random_gaia.csv'), type=str)
    parser.add_argument('--savefig', defaut=None)

    args = parser.parse_args()

    if args.source is not None:
        plot(args.source, background=args.background, savefig=args.savefig)


