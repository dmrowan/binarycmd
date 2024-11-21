#!/usr/bin/env python

import argparse
from astropy import log
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import pandas as pd
from tqdm.autonotebook import tqdm

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

    def __init__(self, Source, mwdust_ext=False, twomass=False):

        self.Source = Source

        self.query_gaia()
        self.query_dist()
        
        if twomass:
            self.query_twomass()

        if mwdust_ext:
            self.query_mwdust()
        self.calculate_color_mags()


    def query_gaia(self):

        r = Vizier(catalog="I/355/gaiadr3",
                   columns=['Source', 'Gmag', 'BP-RP', 'RPlx',
                            'AG', 'E(BP-RP)',
                            'RA_ICRS', 'DE_ICRS',
                            'GLON', 'GLAT']).query_constraints(
                                    Source=str(self.Source))[0]

        r = r.to_pandas().iloc[0].to_dict()

        for k in r.keys():
            if k != 'Source':
                setattr(self, k.replace('-', '_'), r[k])

    def query_twomass(self):
        
        coord = SkyCoord(self.RA_ICRS*u.deg, self.DE_ICRS*u.deg)
        r = Vizier(catalog="II/246/out", 
                   columns=["2MASS", "RAJ2000", "DEJ2000",
                            "Hmag", "Jmag", "Kmag", "+_r"]).query_region(
                                    coord, radius=5*u.arcsec)[0]

        if len(r) == 0:
            log.info('No 2MASS match found')
        elif len(r) == 1:
            r = r.to_pandas().iloc[0].to_dict()
        else:
            r = r.to_pandas().sort_values(by='_r', ascending=True)
            r = r.iloc[0].to_dict()


        setattr(self, 'twomass', r['_2MASS'])
        for k in ['Hmag', 'Jmag', 'Kmag']:
            setattr(self, k, r[k])

        self.J_K = self.Jmag - self.Kmag


    def query_dist(self):

        r = Vizier(catalog="I/352/gedr3dis",
                   columns=['Source', 'rpgeo']).query_constraints(
                        Source=str(self.Source))[0]

        r = r.to_pandas().iloc[0].to_dict()

        self.rpgeo = r['rpgeo']

    def query_mwdust(self):

        from . import get_extinctions

        twomass = hasattr(self, 'twomass')
        r = get_extinctions.evaluate_map(
                self.Source, self.GLON, self.GLAT, self.rpgeo,
                twomass=twomass)

        self.mwdust_av = r[1]
        self.mwdust_ah = r[2]
        self.mwdust_aj = r[3]
        self.mwdust_ak = r[4]

        self.AG = self.mwdust_av*0.789
        self.abp = self.mwdust_av*1.002
        self.arp = self.mwdust_av*0.589
        self.E_BP_RP_ = self.abp-self.arp

        self.E_J_K = self.mwdust_aj - self.mwdust_ak

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

        if hasattr(self, 'twomass'):
            
            self.absolute_k = self.Kmag - 5*np.log10(self.rpgeo)+5 - self.mwdust_ak
            self.j_k_corrected = self.J_K - self.E_J_K


def plot(source_list, twomass=False, 
         ax=None, savefig=None,
         plot_kwargs=None, 
         star_list=None,
         mwdust_ext=False,xlim=None, ylim=None,
         background=get_data_file('random_gaia.csv'),
         hexbin=False,
         save_output=None,
         hexbin_kwargs=None):

    if not cmdutils.check_iter(source_list):

        source_list = [source_list]

    if star_list is None:
        star_list = []
        if len(source_list) > 10:
            iterator = tqdm(source_list)
        else:
            iterator = source_list
        for source in iterator:
            try:
                star = Star(source, mwdust_ext=mwdust_ext, twomass=twomass)
                star_list.append(star)
            except:
                continue
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
        if twomass:
            xlim = (-0.25, 1.2)
        else:
            xlim = (-0.6, 2.4)
    if ylim is None:
        if twomass:
            ylim = (-6, 6)
        else:
            ylim = (-4.1, 8.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()

    if twomass:
        xlabel = r'$J - K$ (mag)'
        ylabel = r'$M_K$ (mag)'
    else:
        xlabel = r'$G_{\rm{BP}}-G_{\rm{RP}}$ (mag)'
        ylabel = r'$M_G$ (mag)'

    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)

    if background is not None:
        df_bkg = cmdutils.pd_read(background)
        if hexbin:
            if hexbin_kwargs is None:
                hexbin_kwargs = {}
            if twomass:
                ax.hexbin(df_bkg.j_k_corrected, df_bkg.absolute_k, **hexbin_kwargs)
            else:
                ax.hexbin(df_bkg.bp_rp_corrected, df_bkg.absolute_g, **hexbin_kwargs)
        else:
            sbkg_kwargs = dict(color='gray', marker='.',
                               edgecolor='none', alpha=0.6,
                               rasterized=True)
            if twomass:
                ax.scatter(df_bkg.j_k_corrected, df_bkg.absolute_k, **sbkg_kwargs)
            else:
                ax.scatter(df_bkg.bp_rp_corrected, df_bkg.absolute_g, 
                           **sbkg_kwargs)

    source = [ s.Source for s in star_list ]
    bp_rp = [ s.bp_rp_corrected for s in star_list ]
    absolute_g = [ s.absolute_g for s in star_list ]
    df_out = pd.DataFrame({'Source':source, 'bp_rp':bp_rp, 'mg':absolute_g})

    if twomass:
        j_k = [ s.j_k_corrected for s in star_list ]
        absolute_k = [ s.absolute_g for s in star_list ]
        df_out['j_k'] = j_k
        df_out['mk'] = absolute_k

        ax.scatter(j_k, absolute_k, **plot_kwargs)
    else:
        ax.scatter(bp_rp, absolute_g, **plot_kwargs)

    if save_output is not None:
        df_out.to_csv(save_output, index=False)

    return plotutils.plt_return(created_fig, fig, ax, savefig)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--source', nargs='+', help='Gaia DR3 Source', default=None)
    parser.add_argument('--background', default=get_data_file('random_gaia.csv'), type=str)
    parser.add_argument('--savefig', defaut=None)
    parser.add_argument('--twomass', default=False, action='store_true')

    args = parser.parse_args()

    if args.source is not None:
        plot(args.source, background=args.background, savefig=args.savefig,
             twomass=args.twomass)


