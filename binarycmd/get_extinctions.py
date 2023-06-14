#!/usr/bin/env python

import argparse
from astropy import log
import multiprocessing as mp
import mwdust
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import cmdutils

log.info('loading in combined19')
combined= mwdust.Combined19(filter='CTIO V')
log.info('loading in combined19 (2mass H)')
combined_2massH = mwdust.Combined19(filter='2MASS H')
log.info('loading in combined19 (2mass J)')
combined_2massJ = mwdust.Combined19(filter='2MASS J')
log.info('loading in combined19 (2mass K)')
combined_2massK = mwdust.Combined19(filter='2MASS Ks')

#Dom Rowan 2021

desc="""
Add mwdust columns to catalog

l and b columns are deg, and distance column is in pc
"""

def add_mwdust(input_catalog, l_column='l', b_column='b',
               distance_column='rpgeo', use_mp=False):

    catalog = cmdutils.pd_read(input_catalog)

    if 'id' not in catalog.columns:
        catalog['id'] = np.arange(len(catalog))
        added_id_col = True
    else:
        added_id_col = False

    catalog['id'] = catalog.id.astype(str)

    for out_col in ['mwdust_av', 'mwdust_ag', 'mwdust_abp', 'mwdust_arp',
                    'mwdust_ah', 'mwdust_aj', 'mwdust_ak']:
        if out_col in catalog.columns:
            raise ValueError("output column already exists in input catalog")

    ag_arr = np.full(len(catalog), np.nan)
    e_bp_rp_arr = np.full(len(catalog), np.nan)

    if use_mp:
        pool = mp.Pool(processes=mp.cpu_count())
        manager = mp.Manager()

        L = manager.list()

        [pool.apply_async(cmdutils.manager_list_wrapper,
                args=(evaluate_map, L,
                      catalog.id.iloc[i],catalog[l_column].iloc[i],
                      catalog[b_column].iloc[i], catalog[distance_column].iloc[i],))
         for i in range(len(catalog))]

        pool.close()
        pool.join()
    else:
        L = [ evaluate_map(catalog.id.iloc[i], catalog[l_column].iloc[i],
                           catalog[b_column].iloc[i], catalog[distance_column].iloc[i])
              for i in tqdm(range(len(catalog))) ]
        

    df_mwdust = pd.DataFrame(list(L))
    df_mwdust.columns = ['id', 'mwdust_av',
                         'mwdust_ah', 'mwdust_aj', 'mwdust_ak']

    df_mwdust['id'] = df_mwdust.id.astype(str)

    catalog = catalog.merge(df_mwdust, on='id', how='left')
    catalog['mwdust_ag'] = catalog.mwdust_av*0.789
    catalog['mwdust_abp'] = catalog.mwdust_av*1.002
    catalog['mwdust_arp'] = catalog.mwdust_av*0.589

    if added_id_col:
        catalog = catalog.drop(columns=['id'])

    if isinstance(input_catalog, str):
        catalog.to_csv(input_catalog, index=False)
    else:
        return catalog

def evaluate_map(id_, l, b, d):

    if not np.isnan(d):
        distance = d/1000
        return (id_, combined(l, b, distance)[0],
                combined_2massH(l,b,distance)[0],
                combined_2massJ(l,b,distance)[0],
                combined_2massK(l,b,distance)[0])
    else:
        return id_, np.nan, np.nan, np.nan, np.nan

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('catalog', type=str)
    parser.add_argument('-l', help = 'galactic longitude column name', type=str, default='l')
    parser.add_argument('-b', help = 'galactic latitude column name', type=str, default='b')
    parser.add_argument('-d', help = 'distance column name', type=str, default='rpgeo')

    args = parser.parse_args()

    add_mwdust(args.catalog, l_column=args.l, b_column=args.b, distance_column=args.d)
