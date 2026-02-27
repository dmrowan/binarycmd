#!/usr/bin/env python

from collections import namedtuple
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d

from . import read_mist_models
from . import cmdutils

#Dom Rowan 2023

desc="""
Functions to read MIST isochrones and evolutionary tracks
and create CMD divisions
"""

mist_iso_results = namedtuple(
        'mist_iso_results', ['ages', 'df', 'sg_spline', 'rg_spline',
                             'color_intersect'])
mist_eep_results = namedtuple(
        'mist_eep_results', ['df', 'sg_spline', 'rg_spline',
                             'color_intersect'])

def mist_iso(path, phot_system):
    
    isocmd = read_mist_models.ISOCMD(path)

    ages = np.round(np.arange(8, 10.25, 0.1,), 1)
    sg_turnoff_vals = np.zeros(len(ages), dtype=object)
    rg_turnoff_vals = np.zeros(len(ages), dtype=object)
                
    df_iso = np.zeros(len(ages), dtype=object)

    for i in range(len(ages)):
                            
        age_idx = isocmd.age_index(ages[i])
        mag0 = isocmd.isocmds[age_idx][phot_system.mist_mag]
        mag1 = isocmd.isocmds[age_idx][phot_system.mist_color0]
        mag2 = isocmd.isocmds[age_idx][phot_system.mist_color1]
        
        #Created binary magnitudes
        flux0 = np.power(10, -0.4*mag0)
        flux1 = np.power(10, -0.4*mag1)
        flux2 = np.power(10, -0.4*mag2)
        mag0_binary = -2.5*np.log10(2*flux0)
        mag1_binary = -2.5*np.log10(2*flux1)
        mag2_binary = -2.5*np.log10(2*flux2)
        
        #Store into pandas df
        df_binary = pd.DataFrame({phot_system.mist_mag:mag0_binary,
                                  phot_system.mist_color0:mag1_binary,
                                  phot_system.mist_color1:mag2_binary})
        df_binary['color'] = (df_binary[phot_system.mist_color0] -
                              df_binary[phot_system.mist_color1])
        df_binary['phase'] = isocmd.isocmds[age_idx]['phase']

        #Identify SG turnoff
        sg_turnoff_vals[i] = (df_binary[df_binary.phase==2].color.iloc[0],
                              df_binary[df_binary.phase==2][phot_system.mist_mag].iloc[0])


        mag0_rg_turnoff = np.max(
                df_binary[
                        (df_binary.phase == 2) &
                        (df_binary.color > phot_system.rg_turnoff_limit)][
                                phot_system.mist_mag])

        rg_turnoff_vals[i] = (df_binary.color.iloc[
                np.where(df_binary[phot_system.mist_mag] == mag0_rg_turnoff)[0][0]],
                              mag0_rg_turnoff)

        if (rg_turnoff_vals[i][1] == df_binary[(df_binary.phase==2) &
            (df_binary.color > phot_system.rg_turnoff_limit)][
                    phot_system.mist_mag].iloc[0]):
            rg_turnoff_vals[i] = (-99,-99)
        else:
            #define the sg phase  == 1
            idx = np.where(
                    (df_binary.phase == 2) &
                    (df_binary.color >= sg_turnoff_vals[i][0]) &
                    (df_binary.color < rg_turnoff_vals[i][0]))[0]
            updated_phase = df_binary.phase.copy().to_numpy()
            updated_phase[idx] = 1
            df_binary['iso_phase'] = updated_phase

        df_iso[i] = df_binary

    #Construct the splines
    sg_spline = interp1d([x[0] for x in sg_turnoff_vals],
                         [x[1] for x in sg_turnoff_vals],
                         kind='linear',
                         fill_value='extrapolate')

    rg_spline = interp1d([x[1] for x in rg_turnoff_vals if x[1] != -99],
                         [x[0] for x in rg_turnoff_vals if x[0] != -99],
                         kind='linear', fill_value='extrapolate')

    sg_points = [(sg_spline.x[-1], sg_spline(sg_spline.x[-1])),
                 (1.5, sg_spline(1.5))]
    rg_points = [(rg_spline(rg_spline.x[-1]), rg_spline.x[-1]),
                 (rg_spline(6), 6)]


    sg_m = (sg_points[1][1]-sg_points[0][1])/(sg_points[1][0]-sg_points[0][0])
    rg_m = (rg_points[1][1]-rg_points[0][1])/(rg_points[1][0]-rg_points[0][0])

    x_intersect = (
            sg_m*sg_points[0][0]-sg_points[0][1]-
            rg_m*rg_points[0][0]+rg_points[0][1])/(sg_m-rg_m)

    return mist_iso_results(ages, df_iso, sg_spline, rg_spline, x_intersect)

def mist_eep(eep_dir, phot_system, iso_path=None, iso_result=None,
             mode='tams', n=1.5):


    eep_files = [ os.path.join(eep_dir, x)
              for x in os.listdir(eep_dir)
              if not x.endswith('eep.cmd') ]
               
    pool = mp.Pool(processes=mp.cpu_count())
    manager = mp.Manager()
    L = manager.list()

    [pool.apply_async(
            cmdutils.manager_list_wrapper_silent,
            args=(eep_worker, L, eep_files[i], eep_files[i]+'.cmd',phot_system, n,),
            kwds=dict(mode=mode))
     for i in range(len(eep_files)) if (
            (os.path.isfile(eep_files[i]+'.cmd')) and
            (1 < path_to_mass(eep_files[i]) < 5))]


    pool.close()
    pool.join()

    df = pd.DataFrame(list(L))

    df = df.sort_values(by='minit', ascending=True).reset_index(drop=True)

    if iso_result is None:
        if iso_path is None:
            raise ValueError('iso path required for eep model')
        iso_result = mist_iso(iso_path, phot_system)

    sg_spline = iso_result.sg_spline
    rg_spline = interp1d(df.color.to_numpy(),
                     df[phot_system.mist_mag].to_numpy(),
                     kind='linear', fill_value='extrapolate')

    #Find the point where sg_spline and rg_spline intersect
    x_intersect = cmdutils.binary_search(
            lambda color: rg_spline(color) - sg_spline(color),
            *phot_system.search_range, epsilon=1e-4, plot=False)

    if np.isnan(x_intersect) or sg_spline(x_intersect) > phot_system.termination_value:
        x_intersect = cmdutils.binary_search(
                lambda color: sg_spline(color) - phot_system.termination_value,
                *phot_system.search_range, epsilon=1e-4, plot=False)


    return mist_eep_results(df, sg_spline, rg_spline, x_intersect)

def eep_worker(eep, eepcmd, phot_system, n, mode='zams'):

    if isinstance(eep, str):
        eep = read_mist_models.EEP(eep, verbose=False)

    if isinstance(eepcmd, str):
        eepcmd = read_mist_models.EEPCMD(eepcmd, verbose=False)

    if eep.minit != eepcmd.minit:
        raise ValueError('evolutionary tracks have different initial masses')

    df_eep = pd.DataFrame({
            'star_age':eep.eeps['star_age'],
            'star_mass':eep.eeps['star_mass'],
            'star_lum':np.power(10, eep.eeps['log_L']),
            'star_radius':np.power(10, eep.eeps['log_R']),
            'phase':eep.eeps['phase']})
    df_eepcmd = pd.DataFrame({
            'star_age':eepcmd.eepcmds['star_age'],
            phot_system.mist_mag:eepcmd.eepcmds[phot_system.mist_mag],
            phot_system.mist_color0:eepcmd.eepcmds[phot_system.mist_color0],
            phot_system.mist_color1:eepcmd.eepcmds[phot_system.mist_color1]})

    df_eepcmd[phot_system.mist_mag+'_binary'] = -2.5*np.log10(
            2*np.power(10, -0.4*df_eepcmd[phot_system.mist_mag]))
    df_eepcmd[phot_system.mist_color0+'_binary'] = -2.5*np.log10(
            2*np.power(10, -0.4*df_eepcmd[phot_system.mist_color0]))
    df_eepcmd[phot_system.mist_color1+'_binary'] = -2.5*np.log10(
            2*np.power(10, -0.4*df_eepcmd[phot_system.mist_color1]))

    df = df_eep.merge(df_eepcmd, on='star_age', how='left')

    idx_zams = np.where(df.phase == 0)[0][0]
    idx_tams = np.where(df.phase == 0)[0][-1]

    if mode == 'zams':
        radius = n*df.star_radius.iloc[idx_zams]
    elif mode == 'tams':
        radius = n*df.star_radius.iloc[idx_tams]
    else:
        raise ValueError(f'invalid mode {mode}')


    df = df[df.phase == 2].reset_index(drop=True)
    idx_end_sg = np.argmin(np.abs(df.star_radius - radius))

    out = {phot_system.mist_mag: df[phot_system.mist_mag+'_binary'].iloc[idx_end_sg],
           'color': (df[phot_system.mist_color0+'_binary'] - df[phot_system.mist_color1+'_binary']).iloc[idx_end_sg],
           'minit':eep.minit}

    return out

def path_to_mass(path):

    path = os.path.split(path)[-1]
    mass_str = path.split('.')[0]
    mass = float(mass_str.rstrip('M'))/1e4

    return mass

