#!/usr/bin/env python

import numpy as np
import pkg_resources
from astroquery.utils.tap.core import TapPlus

from . import get_extinctions

#Dom Rowan 2024

desc="""
Create CSV for random gaia stars w/ 2mass and WISE crossmatch
"""

filename = pkg_resources.resource_filename(__name__, '../data/' + 'random_gaia_test.csv')

def create():

    # Connect to the Vizier TAP service
    tap_service = TapPlus(url="http://tapvizier.cds.unistra.fr/TAPVizieR/tap")

    # Define SQL query
    query = """
    SELECT TOP 50000 gaia.Source, gaia.RPlx, dist.rpgeo, gaia.GLON, gaia.GLAT, 
    gaia.Gmag, gaia.BPmag, gaia.RPmag, gaia."BP-RP", gaia."2MASS", gaia.AllWISE, 
    twomass.Hmag, twomass.Jmag, twomass.Kmag, twomass.Qflg, wise.W1mag, wise.W2mag, wise.qph
    FROM "I/355/gaiadr3" AS gaia
    JOIN "I/352/gedr3dis" AS dist ON gaia.Source = dist.Source
    JOIN "II/246/out" AS twomass ON gaia."2MASS" = twomass."2MASS"
    JOIN "II/328/allwise" AS wise ON gaia.AllWISE = wise.AllWISE
    WHERE gaia.RPlx > 20
    AND mod(gaia.RandomI, 10) = 0
    AND gaia."2MASS" IS NOT NULL
    AND gaia.AllWISE IS NOT NULL
    AND twomass.Qflg = 'AAA'
    AND wise.qph LIKE 'AA%'
    """

    result = tap_service.launch_job_async(query)
    print("SQL Query Completed")

    df = result.get_results().to_pandas()

    df = get_extinctions.add_mwdust(df, l_column='GLON', b_column='GLAT', twomass=True, wise=True)

    #Create columns for absolute mag and color
    df['absolute_g'] = df.Gmag - 5*np.log10(df.rpgeo)+5 - df.mwdust_ag
    df['absolute_k'] = df.Kmag - 5*np.log10(df.rpgeo)+5 - df.mwdust_ak

    df['bp_rp_corrected'] = df['BP-RP'] - (df.mwdust_abp - df.mwdust_arp)
    df['j_k_corrected'] = (df.Jmag - df.Kmag) - (df.mwdust_aj - df.mwdust_ak)
    df['w1_w2_corrected'] = (df.W1mag - df.W2mag) - (df.mwdust_aw1 - df.mwdust_aw2)

    df.to_csv(filename, index=False)


if __name__ == '__main__':
    create()
