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
                      download_dir, ext='eep.cmd'):

    fnames = []

    masses = np.arange(mass_lower, mass_upper+mass_delta, mass_delta)

    n=20

    if len(masses) > n:
        for i in tqdm(range(math.floor(len(masses)/n))):
            
            masses_chunk = masses[i*n:(i+1)*n]

            mass_chunk_lower = masses_chunk[0]
            mass_chunk_upper = masses_chunk[-1]

            mass_chunk_lower = round(mass_chunk_lower, 5)
            mass_chunk_upper = round(mass_chunk_upper, 5)

            fnames.extend(download_mist_eep_worker(mass_chunk_lower, mass_chunk_upper,
                                                   metallicity, mass_delta, download_dir, ext=ext))


        masses_chunk = masses[(i+1)*n:]
        mass_chunk_lower = masses_chunk[0]
        mass_chunk_upper = masses_chunk[-1]

        mass_chunk_lower = round(mass_chunk_lower, 5)
        mass_chunk_upper = round(mass_chunk_upper, 5)

        fnames.extend(download_mist_eep_worker(mass_chunk_lower, mass_chunk_upper,
                                               metallicity, mass_delta, download_dir, ext=ext))
    else:
        
        fnames = download_mist_eep_worker(mass_lower, mass_upper, metallicity, mass_delta, download_dir, ext=ext)
    
    return fnames


def download_mist_eep_worker(mass_lower, mass_upper, metallicity, mass_delta,
                             download_dir, ext='eep.cmd'):

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
               if x.endswith(ext) ]          

    return fnames

