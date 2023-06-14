#!/usr/bin/env python

import numpy as np
import emcee
import corner
import os
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
import pickle
import requests
from scipy.interpolate import interp1d
import zipfile

from . import read_mist_models
from . import get_extinctions
from . import plotutils

"""
params with uncertainties
    - distance
    - Gmag
    - BP mag
    - RP mag

trying to determine mass

sample over age, mass

mass sets the age-(BP_RP) spline, age-MG spline
"""

def download_mist_eep(mass_lower, mass_upper, metallicity, mass_delta,
                      download_dir):

    fnames = []

    masses = np.arange(mass_lower, mass_upper+mass_delta, mass_delta)

    n=20
    for i in tqdm(range(math.floor(len(masses)/n))):
        
        masses_chunk = masses[i*n:(i+1)*n]

        mass_chunk_lower = masses_chunk[0]
        mass_chunk_upper = masses_chunk[-1]

        mass_chunk_lower = round(mass_chunk_lower, 5)
        mass_chunk_upper = round(mass_chunk_upper, 5)

        fnames.extend(download_mist_eep_worker(mass_chunk_lower, mass_chunk_upper,
                                               metallicity, mass_delta, download_dir))


    masses_chunk = masses[(i+1)*n:]
    mass_chunk_lower = masses_chunk[0]
    mass_chunk_upper = masses_chunk[-1]

    mass_chunk_lower = round(mass_chunk_lower, 5)
    mass_chunk_upper = round(mass_chunk_upper, 5)

    fnames.extend(download_mist_eep_worker(mass_chunk_lower, mass_chunk_upper,
                                           metallicity, mass_delta, download_dir))
    
    return fnames


def download_mist_eep_worker(mass_lower, mass_upper, metallicity, mass_delta,
                      download_dir):

    url = "https://waps.cfa.harvard.edu/MIST/track_form.php"
    form_data = {
        "version": "1.2",
        "v_div_vcrit": "vvcrit0.4",
        "mass_value": "",
        "mass_type": "range",
        "mass_range_low": str(mass_lower),
        "mass_range_high": str(mass_upper),
        "mass_range_delta": str(mass_delta),
        "mass_list": "",
        "new_met_value": str(metallicity),
        "output_option": "photometry",
        "output": "UBVRIplus",
        "Av_value": "0"
    }

    # Send the POST request with the form data
    response = requests.post(url, data=form_data)

    # If the response is successful, save the content to a local file
    if response.status_code == 200:
        tmp = response.content.decode().split('"')[1]
        zip_url = f"https://waps.cfa.harvard.edu/MIST/{tmp}"
        zip_path = os.path.join(download_dir, os.path.split(tmp)[1])

        response_zip = requests.get(zip_url)

        # If the response is successful, save the content to a local file
        if response_zip.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(response_zip.content)

    #unzip the file
    extract_dir = os.path.join(download_dir, os.path.splitext(os.path.split(tmp)[1])[0])
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall(extract_dir)

    fnames = [ os.path.join(extract_dir, x) for x in os.listdir(extract_dir)
               if x.endswith('eep.cmd') ]          

    return fnames

def build_spline_df(eep_dir):

    eep_files = [ os.path.join(eep_dir, x) for x in os.listdir(eep_dir)
                  if x.endswith('eep.cmd') ]

    mass = []
    bp_spline = []
    rp_spline = []
    gmag_spline = []
    for fname in tqdm(eep_files):
        
        eepcmd = read_mist_models.EEPCMD(fname, verbose=False)
        df = pd.DataFrame(eepcmd.eepcmds)
        df = df[df.phase == 0].reset_index(drop=True)
        df['star_age'] = df.star_age - df.star_age.min()
        df = df.iloc[1:].reset_index(drop=True)
        df['log_age'] = np.log10(df.star_age)
        df['bpmag'] = df.Gaia_BP_EDR3
        df['rpmag'] = df.Gaia_RP_EDR3
        df['gmag'] = df.Gaia_G_EDR3
        bp_spline_i = interp1d(df.log_age.to_numpy(), df.bpmag.to_numpy(),
                               kind='linear', fill_value='extrapolate')
        rp_spline_i = interp1d(df.log_age.to_numpy(), df.rpmag.to_numpy(),
                               kind='linear', fill_value='extrapolate')
        gmag_spline_i = interp1d(df.log_age.to_numpy(), df.gmag.to_numpy(), 
                                 kind='linear', fill_value='extrapolate')

        mass.append(eepcmd.minit)
        bp_spline.append(bp_spline_i)
        rp_spline.append(rp_spline_i)
        gmag_spline.append(gmag_spline_i)

    df = pd.DataFrame({'mass':mass, 'bp_spline':bp_spline, 'rp_spline':rp_spline, 'gmag_spline':gmag_spline})
    df = df.sort_values(by='mass', ascending=True)

    return df

def main(savefig):


    #J1721 params
    """
    df = build_spline_df('MIST_EEP_feH=-.08')
    bpmag = 13.111749
    rpmag = 12.080743
    gmag = 12.683497
    l = 323.5194137556416
    b = -17.577777568811012
    rpgeo = 250.466873
    rpgeo_b = 249.681244
    rpgeo_B = 251.047714
    """

    #J1208 params
    df = build_spline_df('MIST_EEP_feH=-0.2')
    bpmag = 11.986883
    rpmag = 10.695722
    gmag = 11.422162
    l = 187.1699114247753
    b = 79.701273552841
    rpgeo = 88.6306381
    rpgeo_b = 88.4525528
    rpgeo_B = 88.7761993

    nwalkers = 6
    pos = np.array([0.8, 7, rpgeo])+1e-3*np.random.randn(nwalkers, 3)

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob,
            args=(bpmag, rpmag, gmag, rpgeo, rpgeo_b, rpgeo_B, l, b, df))

    sampler.run_mcmc(pos, 5000, progress=True)

    flat_samples = sampler.get_chain(discard=500, flat=True)

    labels = ['Mass', 'Age', 'Distance']

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    fig = corner.corner(flat_samples, labels=labels, fig=fig, 
                        hist_kwargs=dict(lw=3, color='black'),
                        label_kwargs=dict(fontsize=25), quantiles=[0.5])
    for a in ax.reshape(-1):
        a = plotutils.plotparams(a)

    fig.savefig(savefig)

    with open('mass_test.pickle', 'wb') as p:
        pickle.dump(sampler, p)

def lnprob(theta, bpmag, rpmag, gmag, rpgeo, rpgeo_b, rpgeo_B, l, b, df):

    lp = lnprior(theta, rpgeo, rpgeo_b, rpgeo_B)

    if np.isinf(lp):
        return -np.inf

    like = log_likelihood(theta, bpmag, rpmag, gmag, l, b, df)

    if np.isinf(like):
        return -np.inf

    return lp+like

def lnprior(theta, rpgeo, rpgeo_b, rpgeo_B):

    mass, log_age, dist = theta

    mass_condition = 0.5 <= mass <= 1.2
    age_condition = 6 < log_age < 10

    if mass_condition and age_condition:
        
        rpgeo_sigma = np.max([rpgeo-rpgeo_b, rpgeo_B-rpgeo])
        lp = -0.5*((dist-rpgeo)/rpgeo_sigma)**2

        return lp

    else:
        return -np.inf

def log_likelihood(theta, bpmag, rpmag, gmag, l, b, df):

    mass, log_age, dist = theta

    #Compute extinction
    _, av, _, _, _ = get_extinctions.evaluate_map(0, l, b, dist)
    ag = av*0.789
    abp = av*1.002
    arp = av*0.589

    #calculate corrected bp, rp
    bp_corrected = bpmag - 5*np.log10(dist)+5-abp
    rp_corrected = rpmag - 5*np.log10(dist)+5-arp
    mg = gmag - 5*np.log10(dist)+5-ag

    #get the proper mist splines
    idx = np.argmin(np.abs(df.mass-mass))
    bp_spline = df.bp_spline.iloc[idx]
    rp_spline = df.rp_spline.iloc[idx]
    gmag_spline = df.gmag_spline.iloc[idx]

    mg_mist = gmag_spline(log_age)
    bp_mist = bp_spline(log_age)
    rp_mist = rp_spline(log_age)

    return -0.5*(np.power(mg_mist-mg, 2)+np.power(bp_mist-bp_corrected, 2)+np.power(rp_mist-rp_corrected,2))


if __name__ == '__main__':
   
    main('J1208_cmd_mass_estimation.pdf')

