#!/usr/bin/env python

import argparse
from astropy import log
import multiprocessing as mp
import mwdust
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import cmdutils

dust_maps = {}
shared_dataset = None

#Dom Rowan 2021

desc="""
Add mwdust columns to catalog

l and b columns are deg, and distance column is in pc
"""

def add_mwdust(input_catalog, l_column='l', b_column='b',
               distance_column='rpgeo', use_mp=False,
               twomass=False, wise=False):

    catalog = cmdutils.pd_read(input_catalog)

    load_combined19()

    if twomass:
        load_2mass_H()
        load_2mass_J()
        load_2mass_K()

    if wise:
        load_wise_W1()
        load_wise_W2()
        
    if 'id' not in catalog.columns:
        catalog['id'] = np.arange(len(catalog))
        added_id_col = True
    else:
        added_id_col = False

    catalog['id'] = catalog.id.astype(str)

    for out_col in ['mwdust_av', 'mwdust_ag', 'mwdust_abp', 'mwdust_arp',
                    'mwdust_ah', 'mwdust_aj', 'mwdust_ak',
                    'mwdust_aw1', 'mwdust_aw2']:
        if out_col in catalog.columns:
            raise ValueError("output column already exists in input catalog")

    ag_arr = np.full(len(catalog), np.nan)
    e_bp_rp_arr = np.full(len(catalog), np.nan)

    if use_mp:

        pool = mp.Pool(processes=mp.cpu_count(),
                       initializer=init_pool,
                       initargs=(dust_maps,))
        manager = mp.Manager()

        progress_bar = tqdm(total=len(catalog))

        def callback(result):
            progress_bar.update(1)

        L = manager.list()

        [pool.apply(cmdutils.manager_list_wrapper_silent,
                args=(evaluate_map_mp, L,
                      catalog.id.iloc[i],catalog[l_column].iloc[i],
                      catalog[b_column].iloc[i], catalog[distance_column].iloc[i],
                      twomass,wise),
                callback=callback)
                
         for i in range(len(catalog))]

        pool.close()
        pool.join()
    else:
        L = [ evaluate_map(catalog.id.iloc[i], catalog[l_column].iloc[i],
                           catalog[b_column].iloc[i], catalog[distance_column].iloc[i],
                           twomass=twomass,wise=wise)
              for i in tqdm(range(len(catalog))) ]
        

    df_mwdust = pd.DataFrame(list(L))
    df_mwdust.columns = ['id', 'mwdust_av',
                         'mwdust_ah', 'mwdust_aj', 'mwdust_ak',
                         'mwdust_aw1', 'mwdust_aw2']

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

def evaluate_map(id_, l, b, d, twomass=False, wise=False):

    if 'combined19' not in dust_maps.keys():
        load_combined19()
    
    if not np.isnan(d):
        distance = d/1000

        if twomass:
            if '2MASSH' not in dust_maps.keys():
                load_2mass_H()
                load_2mass_J()
                load_2mass_K()

            mwdust_ah = dust_maps['2MASSH'](l, b, distance)[0]
            mwdust_aj = dust_maps['2MASSJ'](l, b, distance)[0]
            mwdust_ak = dust_maps['2MASSK'](l, b, distance)[0]
        else:
            mwdust_ah = np.nan
            mwdust_aj = np.nan
            mwdust_ak = np.nan

        if wise:
            if 'WISE-1' not in dust_maps.keys():
                load_wise_W1()
                load_wise_W2()

            mwdust_w1=dust_maps['WISE-1'](l, b, distance)[0]
            mwdust_w2=dust_maps['WISE-2'](l, b, distance)[0]
        else:
            mwdust_w1=np.nan
            mwdust_w2=np.nan

        return (id_, dust_maps['combined19'](l, b, distance)[0],
                mwdust_ah,
                mwdust_aj,
                mwdust_ak,
                mwdust_w1,
                mwdust_w2)
    else:
        return id_, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def evaluate_map_mp(id_, l, b, d, twomass=False, wise=False):
    
    if not np.isnan(d):
        distance = d/1000

        if twomass:
            mwdust_ah = shared_dataset['2MASSH'](l, b, distance)[0]
            mwdust_aj = shared_dataset['2MASSJ'](l, b, distance)[0]
            mwdust_ak = shared_dataset['2MASSK'](l, b, distance)[0]
        else:
            mwdust_ah = np.nan
            mwdust_aj = np.nan
            mwdust_ak = np.nan

        if wise:
            mwdust_aw1 = shared_dataset['WISE-1'](l, b, distance)[0]
            mwdust_aw2 = shared_dataset['WISE-2'](l, b, distance)[0]
        else:
            mwdust_aw1 = np.nan
            mwdust_aw2 = np.nan

        return (id_, shared_dataset['combined19'](l, b, distance)[0],
                mwdust_ah,
                mwdust_aj,
                mwdust_ak,
                mwdust_aw1,
                mwdust_aw2)

    else:
        return id_, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def load_combined19():

    if 'combined19' not in dust_maps.keys():
        log.info('loading in combined19')
        combined19 = mwdust.Combined19(filter='CTIO V')

        dust_maps['combined19'] = combined19

def load_2mass_H():
    
    if '2MASSH' not in dust_maps.keys():
        log.info('loading in combined19 2MASS H')
        combined_2massH = mwdust.Combined19(filter='2MASS H')

        dust_maps['2MASSH'] = combined_2massH

def load_2mass_J():

    if '2MASSJ' not in dust_maps.keys():
        log.info('loading in combined19 2MASS J')
        combined_2massJ = mwdust.Combined19(filter='2MASS J')

        dust_maps['2MASSJ'] = combined_2massJ

def load_2mass_K():

    if '2MASSK' not in dust_maps.keys(): 
        log.info('loading in combined19 2MASS K')
        combined_2massK = mwdust.Combined19(filter='2MASS Ks')

        dust_maps['2MASSK'] = combined_2massK

def load_wise_W1():

    if 'WISE-1' not in dust_maps.keys(): 
        log.info('loading in combined19 WISE-1')
        combined_wise1 = mwdust.Combined19(filter='WISE-1')

        dust_maps['WISE-1'] = combined_wise1

def load_wise_W2():

    if 'WISE-2' not in dust_maps.keys(): 
        log.info('loading in combined19 WISE-2')
        combined_wise2 = mwdust.Combined19(filter='WISE-2')

        dust_maps['WISE-2'] = combined_wise2

def init_pool(dataset):
    
    global shared_dataset
    shared_dataset = dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('catalog', type=str)
    parser.add_argument('-l', help = 'galactic longitude column name', type=str, default='l')
    parser.add_argument('-b', help = 'galactic latitude column name', type=str, default='b')
    parser.add_argument('-d', help = 'distance column name', type=str, default='rpgeo')
    parser.add_argument('--twomass', default=False, action='store_true')
    parser.add_argument('--wise', default=False, action='store_true')
    parser.add_argument('--mp', default=False, action='store_true')

    args = parser.parse_args()

    add_mwdust(args.catalog, l_column=args.l, b_column=args.b, distance_column=args.d,
               twomass=args.twomass, wise=args.wise, use_mp=args.mp)
