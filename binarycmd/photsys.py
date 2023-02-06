#!/usr/bin/env python

#Dom Rowan 2023


class PhotSystem:

    def __init__(self, mag=None, absolute_mag=None, color=None,
                 color0=None, color1=None,
                 mist_mag=None, mist_color0=None, mist_color1=None,
                 termination_value=None, rg_turnoff_limit=None,
                 xlabel='Color (mag)', ylabel=r'$M$ (mag)',
                 plt_range=None):


        self.mag = mag
        self.absolute_mag = absolute_mag
        self.color = color
        self.color_corrected = f'{self.color}_corrected'
        self.color0 = color0
        self.color1 = color1
        self.mist_mag = mist_mag
        self.mist_color0 = mist_color0
        self.mist_color1 = mist_color1
        self.termination_value = termination_value
        self.rg_turnoff_limit = rg_turnoff_limit

        self.xlabel = xlabel
        self.ylabel = ylabel

        if plt_range is None:
            plt_range = [(-0.6, 2.5), (-4, 8.5)]

        self.plt_range = plt_range


    @classmethod
    def gaia_dr3(cls):

        mag = 'Gmag'
        absolute_mag = 'absolute_g'
        color = 'BP-RP'
        color0 = 'BPmag'
        color1 = 'RPmag'

        mwdust_mag = 'mwdust_ag'
        mwdust_c0 = 'mwdust_abp'
        mwdust_c1 = 'mwdust_arp'

        mist_mag = 'Gaia_G_EDR3'
        mist_color0 = 'Gaia_BP_EDR3'
        mist_color1 = 'Gaia_RP_EDR3'

        termination_value = 4.5

        rg_turnoff_limit = 0.9

        xlabel = r'$G_{\rm{BP}}-G_{\rm{RP}}$ (mag)'
        ylabel = r'$M_G$ (mag)'

        plt_range = [(-0.6, 2.4), (-4.1, 8.5)]

        return cls(mag=mag, absolute_mag=absolute_mag,
                   color=color, mist_mag=mist_mag,
                   mist_color0=mist_color0, mist_color1=mist_color1,
                   termination_value=termination_value,
                   rg_turnoff_limit=rg_turnoff_limit,
                   xlabel=xlabel, ylabel=ylabel, plt_range=plt_range)

    @classmethod
    def gaia_default(cls):

        mag = 'phot_g_mean_mag'
        absolute_mag = 'absolute_g'
        color = 'bp_rp'

        mist_mag = 'Gaia_G_EDR3'
        mist_color0 = 'Gaia_BP_EDR3'
        mist_color1 = 'Gaia_RP_EDR3'

        termination_value = 4.5

        rg_turnoff_limit = 0.9

        xlabel = r'$G_{\rm{BP}}-G_{\rm{RP}}$ (mag)'
        ylabel = r'$M_G$ (mag)'

        #plt_range = [(-0.6, 2.5), (-4, 8.5)]
        plt_range = [(-0.6, 2.4), (-4.1, 8.5)]

        return cls(mag=mag, absolute_mag=absolute_mag,
                   color=color, mist_mag=mist_mag,
                   mist_color0=mist_color0, mist_color1=mist_color1,
                   termination_value=termination_value,
                   rg_turnoff_limit=rg_turnoff_limit,
                   xlabel=xlabel, ylabel=ylabel, plt_range=plt_range)


    @classmethod
    def twomass_default(cls):

        mag = 'kmag'
        absolute_mag = 'absolute_k'
        color = 'j_k'

        mist_mag = '2MASS_Ks'
        mist_color0 = '2MASS_J'
        mist_color1 = '2MASS_Ks'

        termination_value = 3

        rg_turnoff_limit = 0.4

        xlabel = r'$J - K$ (mag)'
        ylabel = r'$M_K$ (mag)'

        plt_range = [ (-0.25, 1.2), (6, -6) ]

        return cls(mag=mag, absolute_mag=absolute_mag,
                   color=color, mist_mag=mist_mag,
                   mist_color0=mist_color0, mist_color1=mist_color1,
                   termination_value=termination_value,
                   rg_turnoff_limit=rg_turnoff_limit,
                   xlabel=xlabel, ylabel=ylabel, plt_range=plt_range)


