import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from scipy.ndimage import gaussian_filter
from lmfit import Parameters,minimize, fit_report
from matplotlib.widgets import SpanSelector, Button
import time
import scipy
from typing import List, Dict, Union
from IonClass import *
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# from scipy import *
# import ndimage
CSPEED_KMS = 299989. #speed of light
CSPEED_CGS = 2.997925e10


N_ELEMENTS = 24
N_LVL = 6
ELEMENTS = {
'H' : 1,
'O' : 2,
'C' : 3,
'S' : 4,
'Si': 5,
'N' : 6,
'Mg': 7,
'P' : 8,
'Fe': 9,
'Ni': 10,
'Ca': 11,
'Cl': 12,
'Co': 13,
'Cr': 14,
'Cu': 15,
'Al': 16,
'Ar': 17,
'F' : 18,
'Ga': 19,
'Ge': 20,
'Mn': 21,
'Ti':22,
'V': 23,
'Zn': 24
}

ION_LVL = {
'I' : 1,
'II' :2,
'III': 3,
'IV':4,
'V': 5,
'VI':6
}

COLORS = [
'#6a3d9a',
'#ffff99',
'#8dd3c7',
'#ffffb3',
'#a6cee3',
'#1f78b4',
'#b2df8a',
'#33a02c',
'#fb9a99',
'#bc80bd',
'#fdbf6f',
'#ff7f00',
'#cab2d6',
'#bebada'
]

class SpecData:
    def __init__(self, wave_arr, flux_arr, err_arr = None, redshift = None, vdisp = 20):
        self.wave          = wave_arr
        self.flux_cont_div = flux_arr
        self.err_cont_div  = err_arr if err_arr is not None else np.ones(len(wave_arr))
        self.vdisp         = vdisp
        self.z             = redshift
        self.masks         = []

    def get_good_arr(self):
        good = np.any([(self.wave > window[0]) & (self.wave < window[1]) for window in self.masks], axis=0)
        ydataAbs = self.flux_cont_div[good]
        xdataAbs = self.wave[good]
        yerrAbs  = self.err_cont_div[good]
        return [xdataAbs, ydataAbs, yerrAbs]

class FitController:
    def __init__(self, wave_arr: np.ndarray, flux_arr: np.ndarray, err_arr: np.ndarray = None, redshift: float = None, vdisp: float = 20) -> None:
        self.spectrum = SpecData(wave_arr, flux_arr, err_arr, redshift, vdisp)
        self.linelist         =  pd.read_csv('linelist.csv')
        if(redshift):
            self.linelist['MW_Wave']  = self.linelist['Wave']/(1+redshift)
        self.model    = AbsLineModel(wave_arr, vdisp, self.linelist)

    def include_lines(self, line_names: List[str]) -> None:
        """Include a list of spectral lines to be fit in the spectrum.
        Args:
            line_names (List[str]): List of line names to be included.

        Returns:
            None: This method doesn't return anything, it just updates the state of the object.

        Notes:
            This method updates the line fitting model stored in the `model` attribute of the object. It includes the lines
            listed in `line_names`, sets the 'Fit' column of the corresponding lines in the `linelist` attribute of the `model`
            to 1, and initializes the parameters for each ion.
        """
        if(self.model.include_lines(line_names)):
            self.model.print_lines_to_fit()
            print('Initializing the parameters for each ion')
            self.model.initialize_parameters()


    def define_masks(self, masks = None, interactive = 0):
        self.spectrum.masks = self.model.define_masks(self.spectrum.wave, self.spectrum.flux_cont_div, masks, interactive)
        print('Showing the spectrum with masks')
        self.plot_spectra(masks = True)

    def include_MW_lines(self, line_names = None):
        """Include a list of spectral lines to be fit in the spectrum.
        Args:
            line_names (List[str]): List of line names to be included.

        Returns:
            None: This method doesn't return anything, it just updates the state of the object.

        Notes:
            This method updates the line fitting model stored in the `model` attribute of the object. It includes the lines
            listed in `line_names`, sets the 'Fit' column of the corresponding lines in the `linelist` attribute of the `model`
            to 1, and initializes the parameters for each ion.
        """
        if(line_names):
            ierr = self.model.include_MW_lines(self.spectrum.masks, self.spectrum.z, line_names)
        else:
            ierr = self.model.include_MW_lines(self.spectrum.masks, self.spectrum.z)

        if(ierr):
            self.model.print_MW_lines_to_fit(self.spectrum.z)
            print('Initializing the parameters for each MW ion')
            self.model.initialize_MW_parameters()

    def print_lines(self) -> None:
        print('------------ List of lines which can be included --------------')
        for i, element in enumerate(ELEMENTS.keys()):
                for j, ion_level in enumerate(ION_LVL.keys()):
                        temp = self.linelist[(self.linelist['Elt'] == element) &
                                                   (self.linelist['Lvl'] == ion_level) &
                                                                              (self.linelist['A'] >1)]
                        if(len(temp)==0):
                            continue
                        print(f'\t -- {element} {ion_level} ;')

                        print(temp[['Name', 'Wave', 'f', 'A']].to_string(index=False))
                        print('-----------')



    def plot_spectra(self, masks=False, fit=False, fit_ion=False, fit_MW_ion = False):
        """
        Plots the normalized spectrum, absorption lines and optionally fits and individual ions.

        Parameters
        ----------
        masks : bool, optional
            If True, masks out the regions outside the defined masks. Default is False.
        fit : bool, optional
            If True, plot the fitted spectrum. Default is False.
        fit_ion : bool, optional
            If True, plot individual ions for each absorption line. Default is False.

        Returns
        -------
        None.

        """
        if masks: # Define the masked arrays
            good = np.any([(self.spectrum.wave > window[0]) & (self.spectrum.wave < window[1]) for window in self.spectrum.masks], axis=0)

            ydataAbs = self.spectrum.flux_cont_div[good]
            xdataAbs = self.spectrum.wave[good]
            yerrAbs  = self.spectrum.err_cont_div[good]
            pos = np.where(np.abs(np.diff(xdataAbs)) >= self.spectrum.wave[2]-self.spectrum.wave[0])[0]+1
            # pos      = np.where(np.abs(np.diff(xdataAbs)) >= self.wave[2]-self.wave[0])[0]+1
            ydataAbs = np.insert(ydataAbs, pos, np.nan)
            xdataAbs = np.insert(xdataAbs, pos, np.nan)
            yerrAbs  = np.insert(yerrAbs, pos, np.nan)

        # FInd optimal number of subplots
        temp = self.model.linelist[self.model.linelist['Fit'] == 1].sort_values('Wave').reset_index(drop=True)
        groups = []
        groups_name = []
        if(len(temp)>1):
            cur_group = [temp['Wave'][0]]
            cur_group_name = [temp['Name'][0]]

            comp_i = 0
            for i in range(1, len(temp)):
                if temp['Wave'][i] - cur_group[0] <= 2:
                    cur_group.append(temp['Wave'][i])
                    cur_group_name.append(temp['Name'][i])
                else:
                    groups.append(cur_group)
                    groups_name.append(cur_group_name)
                    cur_group = [temp['Wave'][i]]
                    cur_group_name =[temp['Name'][i]]
            groups.append(cur_group)
            groups_name.append(cur_group_name)
        n_plots = len(groups)


        #Producing the final spectrum:
        if(fit):
            final_spectrum = self.model.fit
            # final_spectrum = np.ones(len(self.wave))
            # for ion in self.ions_list:
            #     final_spectrum *= ion.spectrum

        ## Plotting the main spectrum
        fig = plt.figure(figsize=(20, int(np.ceil(n_plots/3.)*6)+6))
        gs = GridSpec(int(np.ceil(n_plots/3.))+1, 3, figure=fig)

        ax = fig.add_subplot(gs[0, :])
        ax.plot(self.spectrum.wave, self.spectrum.flux_cont_div, linewidth = 2, color='k', drawstyle='steps-mid')
        ax.fill_between(self.spectrum.wave, y1 = self.spectrum.flux_cont_div+self.spectrum.err_cont_div,
                                            y2 = self.spectrum.flux_cont_div-self.spectrum.err_cont_div,
                        color='gray', alpha = 0.2)
        if(masks):
            ax.plot(xdataAbs, ydataAbs, linewidth=2, color='g', drawstyle='steps-mid')
            ax.set_xlim(int(np.nanmax([np.nanmin(self.spectrum.masks) - 10, np.nanmin(self.spectrum.wave)])),
                        int(np.nanmin([np.nanmax(self.spectrum.masks)+ 10, np.nanmax(self.spectrum.wave)])))
        else:
            ax.set_xlim(np.nanmin(self.spectrum.wave),np.nanmax(self.spectrum.wave))

        if(fit):
            ax.plot(self.spectrum.wave, final_spectrum, linewidth = 2, color='royalblue', drawstyle='steps-mid')
        ax.set_ylim(0,2)
        ax.set_title('Overview', fontsize=17)
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=17)
        ax.set_ylabel(r'Normalized Flux', fontsize=17)




        # create manual symbols for legend
        patch = mpatches.Patch(color='grey', label='Uncertainty')
        line = []
        line.append(Line2D([0], [0], label='Data', color='k'))
        line.append(patch)
        if(fit):
            line.append(Line2D([0], [0], label='Combined fit', color='royalblue'))
        if masks:
            line.append(Line2D([0], [0], label='Constraints', color='g'))
        if(fit_ion):
            for ion_df in self.model.ions_list:
                line.append(Line2D([0], [0], label=ion_df.name, color=ion_df.color, ls = '--'))
        if fit_MW_ion:
                line.append(Line2D([0], [0], label='MW lines', color='r', ls = ':'))



        ax.legend(handles=line, loc = (1.01,0))

            #Plotting individual windows
        for i, group in enumerate(groups):
            ax = fig.add_subplot(gs[int(1+i/3), int(i%3)])
            ax.plot(self.spectrum.wave, self.spectrum.flux_cont_div, linewidth=2, color='k', drawstyle='steps-mid')
            ax.fill_between(self.spectrum.wave, y1=self.spectrum.flux_cont_div+self.spectrum.err_cont_div,
                                                y2=self.spectrum.flux_cont_div-self.spectrum.err_cont_div,
                            color='gray', alpha=0.2)
            ax.plot(xdataAbs, ydataAbs, linewidth=2, color='g', drawstyle='steps-mid')
            if(fit):
                ax.plot(self.spectrum.wave, final_spectrum, linewidth=2, color='royalblue', drawstyle='steps-mid')
            if fit_ion:
                for ion_df in self.model.ions_list:
                    ax.plot(self.spectrum.wave, ion_df.spectrum, linewidth=1, drawstyle='steps-mid', ls='--', c =ion_df.color)

            if fit_MW_ion:
                all_MW_spec = np.ones(len(self.spectrum.wave))
                for ion_df in self.model.ions_MW_list:
                    all_MW_spec *= ion_df.spectrum
                ax.plot(self.spectrum.wave, all_MW_spec, linewidth=2, drawstyle='steps-mid', ls=':', c= 'r', alpha = 0.7)

            ax.set_ylim(0, 2)
            mean_w = np.mean(group)
            ax.set_xlim(mean_w - 5, mean_w + 5)
            if('HI1216' in groups_name[i]):
                ax.set_xlim(mean_w - 15, mean_w + 15)
            title = ' + '.join(groups_name[i])
            for w in group:
                ax.axvline(w, 0.8, 1, ls=':')
            ax.set_title(title, fontsize=16)
        fig.subplots_adjust(wspace=0.3, hspace = 0.3)
        return fig


    def plot_lines(self,  elt_list, MW = True, f = 1e-4, xbounds = None,):

        # FInd optimal number of subplots
        # print(elt_list)

        temp = self.model.linelist[(self.model.linelist['f'] > f) & (self.model.linelist['Ion'].isin(elt_list))].sort_values('Wave').reset_index(drop=True)
        tempMW = self.model.linelist[(self.model.linelist['f'] > f) & self.model.linelist['MW'] == 1].sort_values('Wave').reset_index(drop=True)

        if xbounds == None:
            xbounds = [np.nanmin(self.spectrum.wave), np.nanmax(self.spectrum.wave)]
        xrange = xbounds[1]-xbounds[0]
        nplots = int(np.ceil(xrange/50))

        fig = plt.figure(figsize=(20, int(np.ceil(nplots)*6)))
        gs = GridSpec(int(np.ceil(nplots)), 1, figure=fig)

        for i in range(nplots):

            ax = fig.add_subplot(gs[int(i)])
            ax.plot(self.spectrum.wave, self.spectrum.flux_cont_div, linewidth = 2, color='k', drawstyle='steps-mid')
            ax.fill_between(self.spectrum.wave, y1 = self.spectrum.flux_cont_div+self.spectrum.err_cont_div,
                                                y2 = self.spectrum.flux_cont_div-self.spectrum.err_cont_div,
                            color='gray', alpha = 0.2)

            ax.set_xlim(np.nanmin(self.spectrum.wave)+50*i,np.nanmin(self.spectrum.wave)+50*(i+1))
            ax.set_ylim(0,2)

            temp2 = temp[(temp['Wave'] > np.nanmin(self.spectrum.wave)+50*i) & (temp['Wave']<= np.nanmin(self.spectrum.wave)+50*(i+1))]
            # print(temp2)
            for ii, row in temp2.iterrows():
                ax.axvline(row.Wave, ymin = 0.9, ymax =1, lw = 2, color = 'royalblue')
                ax.text(row.Wave, 0.8*2, row.Ion, rotation = 90, horizontalalignment = 'center', c= 'royalblue', fontsize = 15,
                   bbox={'facecolor':'white','alpha':1,'edgecolor':'black'})#                    bbox={'facecolor':'white','alpha':1,'edgecolor':'black','pad':1})

            if(MW):
                # print(self.model.linelist[(self.model.linelist['f'] > f) & self.model.linelist['MW'] == 1])
                temp2 = tempMW[(np.nanmin(self.spectrum.wave)+50*(i) < tempMW['MW_Wave']) & (tempMW['MW_Wave']< np.nanmin(self.spectrum.wave)+50*(i+1)  )]

                for i, row in temp2.iterrows():
                    ax.axvline(row.MW_Wave, ymin = 0.9, ymax =1, lw = 2, color = 'Firebrick')
                    ax.text(row.MW_Wave, 0.7*2, 'MW'+row.Ion, rotation = 90, horizontalalignment = 'center', c= 'Firebrick', fontsize = 15,
                       bbox={'facecolor':'white','alpha':1,'edgecolor':'black'})#
            # if(fit):
            #     ax.plot(self.spectrum.wave, final_spectrum, linewidth = 2, color='royalblue', drawstyle='steps-mid')
            # ax.set_ylim(0,2)
            # ax.set_title('Overview', fontsize=17)
            # ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=17)
            # ax.set_ylabel(r'Normalized Flux', fontsize=17)
            #     #Plotting individual windows
            # for i, group in enumerate(groups):
            #     ax = fig.add_subplot(gs[int(1+i/3), int(i%3)])
            #     ax.plot(self.spectrum.wave, self.spectrum.flux_cont_div, linewidth=2, color='k', drawstyle='steps-mid')
            #     ax.fill_between(self.spectrum.wave, y1=self.spectrum.flux_cont_div+self.spectrum.err_cont_div,
            #                                         y2=self.spectrum.flux_cont_div-self.spectrum.err_cont_div,
            #                     color='gray', alpha=0.2)
            #     ax.plot(xdataAbs, ydataAbs, linewidth=2, color='g', drawstyle='steps-mid')
            #     if(fit):
            #         ax.plot(self.spectrum.wave, final_spectrum, linewidth=2, color='royalblue', drawstyle='steps-mid')
            #     if fit_ion:
            #         for ion_df in self.model.ions_list:
            #             ax.plot(self.spectrum.wave, ion_df.spectrum, linewidth=2, drawstyle='steps-mid', ls=':')
            #
            #     if fit_MW_ion:
            #         for ion_df in self.model.ions_MW_list:
            #             ax.plot(self.spectrum.wave, ion_df.spectrum, linewidth=2, drawstyle='steps-mid', ls='--', c= 'r')
            #
            #     ax.set_ylim(0, 2)
            #     mean_w = np.mean(group)
            #     ax.set_xlim(mean_w - 5, mean_w + 5)
            #     title = ' + '.join(groups_name[i])
            #     for w in group:
            #         ax.axvline(w, 0.8, 1, ls=':')
            #     ax.set_title(title, fontsize=16)
        fig.subplots_adjust(wspace=0.3, hspace = 0.3)
        return fig

    def update_parameter(self, name, value = None, minV = None, maxV = None, expr = None, vary = None):
        self.model.update_parameter(name,  value,  minV,  maxV,  expr, vary)

    def fit_model(self, method = 'ampgo') -> None:
        """
        Fits the specified absorption line model to the spectrum data.

        Args:
            None

        Returns:
            None

        """
        x, y, yerr = self.spectrum.get_good_arr()
        self.model.run_fit(x, y, yerr, method = method)

        print('Ion parameters from fit')
        for ion in self.model.ions_list:
            ion.print_ion_parameters()
            print()

        print('MW Ion parameters from fit')
        for ion in self.model.ions_MW_list:
            ion.print_ion_parameters()
            print()

        print('Plotting the final fit')
        self.plot_spectra(masks=True, fit=True, fit_ion=True, fit_MW_ion = True)


    def get_spectra(self):
        """
        Retrieves the spectrum data and fitted model components.

        Args:
            None

        Returns:
            A dictionary containing the following keys and values:
                - 'wave': an array of wavelengths in Angstroms
                - 'flux': an array of flux values, continuum-divided
                - 'err': an array of flux errors, continuum-divided
                - 'all': an array of the full fitted model (continuum + absorption lines)
                - {ion name}: a dictionary containing the fitted absorption line flux values for a particular ion, continuum-divided
        """
        spectra_dict = {'wave': self.spectrum.wave,
                'flux': self.spectrum.flux_cont_div,
                'err' : self.spectrum.err_cont_div,
                'all' : self.model.fit}
        for ion in self.model.ions_list:
            spectra_dict[ion.name] = ion.spectrum

        for ion in self.model.ions_MW_list:
            spectra_dict['MW'+ion.name] = ion.spectrum
        return spectra_dict

    def get_parameters(self) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Retrieves the fitted parameters for each ion.

        Args:
            None

        Returns:
            A dictionary containing the following keys and values:
                - {ion name}: a dictionary containing the following keys and values:
                    - 'v': the fitted velocity in km/s
                    - 'v_err': the error in the fitted velocity in km/s
                    - 'b': the fitted Doppler broadening parameter in km/s
                    - 'b_err': the error in the fitted Doppler broadening parameter in km/s
                    - 'logn': the fitted column density in cm^-2
                    - 'logn_err': the error in the fitted column density in cm^-2
                    - 'fcov': the fitted covering fraction
                    - 'fcov_err': the error in the fitted covering fraction

        """
        parameters_dict = {}

        # Add the fitted parameters for each ion
        for ion in self.model.ions_list:
            params_dict = {'v': ion.v[0],
                           'v_err': ion.v[1],
                           'b': ion.b[0],
                           'b_err': ion.b[1],
                           'logn': ion.logn[0],
                           'logn_err': ion.logn[1],
                           'fcov': ion.fcov[0],
                           'fcov_err': ion.fcov[1]}
            parameters_dict[ion.name] = params_dict

        return parameters_dict



class AbsLineModel:
    def __init__(self, wave_arr, vdisp, linelist):
        self.wave             = wave_arr
        self.ions_fit         = np.zeros((N_ELEMENTS, N_LVL))
        self.ions_MW_fit      = np.zeros((N_ELEMENTS, N_LVL))
        self.linelist         =  linelist
        self.linelist['Fit']     = 0
        self.linelist['MW_Fit']  = 0
        self.MW_fit           = 0
        self.fit              = np.zeros(len(wave_arr))
        self.init_parameters  = None
        self.final_parameters = None
        self.init_MW_parameters  = None
        self.final_MW_parameters = None
        self.vdisp            = vdisp

    def H(self, a, x):
        P = x**2
        H0 = np.exp(-x**2)
        Q = 1.5/x**2
        return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)

    def voigt_profile(self, xl, vout, b, log_n, wave0, f, A):

        c = 2.99792e10        # cgs#cm/s
        m_e = 9.1094e-28       # g
        e = 4.8032e-10        # cgs units
        cspeed = c/1e5
        x0 = wave0*((vout*1e5)/c + 1.) #This is the wavelength from line center in Angstroms
        C_a = np.sqrt(np.pi)*e**2*f*wave0*1.e-8/m_e/c/(b*1e5)
        a = wave0*1.e-8*A/(4.*np.pi*b*1e5)
        x = (cspeed / b) * (1. - x0/xl)
        N = 10**log_n
        tau = np.float64(C_a) * N * self.H(a, x)
        flux = np.exp(-tau) #This converts the optical depth to Flux units using the radiative transfer equation
        sigma_pixel = self.vdisp / ((xl[1]-xl[0])*cspeed/wave0) #& $  ; gaussian sigma of convolution kernel in pixels
        cflux = gaussian_filter(flux, sigma = sigma_pixel) #& $ ;Convolve to the resolution of the data.
        return cflux

    #
    #
    # def tau_line(self, l: float, f: float, gamma: float, b_km: float, wave: np.ndarray = None) -> np.ndarray:
    #     """
    #     Compute the optical depth (tau) of a spectral line as a function of wavelength.
    #
    #     Parameters
    #     ----------
    #     l : float
    #         Wavelength (in Angstroms) of the line center.
    #     f : float
    #         Oscillator strength of the transition.
    #     gamma : float
    #         Damping constant (in s^-1) of the transition.
    #     b_km : float
    #         Doppler broadening parameter (in km/s).
    #     wave : np.ndarray, optional
    #         Wavelength grid (in Angstroms) to compute the tau on. If not provided, a grid will be automatically generated.
    #
    #     Returns
    #     -------
    #     np.ndarray
    #         A 2D array of shape (N, 2) where N is the number of wavelength points. The first column contains the
    #         wavelength values in Angstroms and the second column contains the corresponding optical depths.
    #
    #     """
    #
    #     lam = l * 1.e-8
    #     nu = CSPEED_CGS / lam
    #     dnu = nu / CSPEED_CGS * b_km * 1.e5
    #     a = gamma / (4. * np.pi * dnu)
    #
    #     if wave is None or len(wave) == 0:
    #         print('Computing wavelength grid...')
    #         lmin = lam - CSPEED_CGS / nu ** 2 * (np.sqrt(2.654e-2 * f * gamma * nu / CSPEED_CGS) * 5e-4 + 10. * dnu)
    #         lmax = lam + CSPEED_CGS / nu ** 2 * (np.sqrt(2.654e-2 * f * gamma * nu / CSPEED_CGS) * 5e-4 + 10. * dnu)
    #         bin = 0.0005e-8
    #         wave = np.arange(lmin, lmax, bin)
    #
    #     lams = wave * 1.e-8
    #     freq = CSPEED_CGS / lams
    #     u_range = (freq - nu) / dnu
    #     gammaLor = gamma * lam ** 2 / (2 * np.pi * CSPEED_CGS)
    #     sigmaGauss = dnu
    #     x = u_range * np.sqrt(2) * sigmaGauss
    #     h = scipy.special.voigt_profile(x, sigmaGauss, gammaLor) * np.sqrt(2) * np.sqrt(np.pi) * sigmaGauss  # Real equation
    #     sig = 2.654e-2 * f * h / dnu / np.sqrt(np.pi)
    #     out = np.column_stack((lams * 1.e8, sig))
    #     return out
    #
    # def voigt_profile(self, x, v, b, log_n, l0, f, A):
    #     """
    #     Compute the Voigt line profile.
    #
    #     Parameters
    #     ----------
    #     x : array_like
    #         Wavelength grid over which to compute the line profile.
    #     v : float
    #         Velocity shift of the line profile in km/s.
    #     b : float
    #         Doppler broadening parameter of the line profile in km/s.
    #     log_n : float
    #         Logarithm of the column density in cm^-2.
    #     l0 : float
    #         Rest wavelength of the transition in Angstroms.
    #     f : float
    #         Oscillator strength of the transition.
    #     A : float
    #         Einstein A coefficient of the transition in s^-1.
    #     vdisp : float
    #         Velocity dispersion of the data in km/s.
    #
    #     Returns
    #     -------
    #     voigt : `~numpy.ndarray`
    #         The computed Voigt line profile.
    #     """
    #     tau     = 0. #The initial Ly-Beta optical depth is zero.
    #     l_shift = l0*(v/CSPEED_KMS + 1.) #This is the wavelength from line center in Angstroms
    #     tau_arr = self.tau_line(l_shift, f, A, b, x) #This calls a function which calculates the line profile using the f-value and Einstein A coefficient of Ly-B
    #     N_line  = 10.**log_n #need column density in normal units
    #     tau     = tau + (N_line * tau_arr[:,1]) #calculate the optical depth of the line profile
    #     flux    = np.exp(-tau) #This converts the optical depth to Flux units using the radiative transfer equation
    #     sigma_pixel = self.vdisp / ((x[1]-x[0])*CSPEED_KMS/l0) #& $  ; gaussian sigma of convolution kernel in pixels
    #     voigt   = gaussian_filter(flux, sigma = sigma_pixel) #& $ ;Convolve to the resolution of the data.
    #     return voigt



    def include_lines(self, line_names):

        self.linelist['Fit'] = 0
        self.ions_fit *= 0
        self.ions_list = []

        for line_name in line_names:
            if line_name not in self.linelist['Name'].values:
                print(f'{line_name} does not exist in the csv. Please choose among these lines')
                print()
                for Elt in ELEMENTS:
                    dfElt = self.linelist.loc[self.linelist['Elt'] == Elt]
                    print(dfElt['Name'].values)
                # print(self.linelist['Name'])
                return 0


            line_data = self.linelist.loc[self.linelist['Name'] == line_name]
            elt_index = ELEMENTS[line_data['Elt'].values[0]] - 1
            lvl_index = ION_LVL[line_data['Lvl'].values[0]] - 1

            if not self.ions_fit[elt_index, lvl_index]:
                ion = Ion(line_data['Elt'].values[0] + line_data['Lvl'].values[0], len(self.wave), COLORS[len(self.ions_list)])
                self.ions_list.append(ion)

            self.ions_fit[elt_index, lvl_index] += 1
            self.linelist.loc[self.linelist['Name'] == line_name, 'Fit'] = 1
            self.ions_list[-1].list_lines.append(line_name)

        return 1



    def print_lines_to_fit(self) -> None:
        print('------------ List of lines included in the fit --------------')
        for i, element in enumerate(ELEMENTS.keys()):
            n_ions_fit = np.sum(self.ions_fit[i])
            if n_ions_fit > 0:
                print(f' {element} with {n_ions_fit:.0f} line(s) included:')
                for j, ion_level in enumerate(ION_LVL.keys()):
                    if self.ions_fit[i][j] > 0:
                        print(f'\t -- {element} {ion_level} ;')
                        temp = self.linelist[(self.linelist['Elt'] == element) &
                                                   (self.linelist['Lvl'] == ion_level) &
                                                   (self.linelist['Fit'] == 1)]
                        print(temp[['Name', 'Wave', 'f', 'A']].to_string(index=False))
                        print('-----------')




    def include_MW_lines(self, masks, z, line_names = None):

        self.ions_MW_fit *= 0
        self.ions_MW_list = []
        self.linelist['MW_Fit'] = 0


        if(line_names is not None):
            #self.linelist['MW'] = 0
            for line_name in line_names:
                if line_name not in self.linelist['Name'].values:
                    print(f'{line_name} does not exist in the csv. Please choose among these lines')
                    print()
                    for Elt in ELEMENTS:
                        dfElt = self.linelist.loc[self.linelist['Elt'] == Elt]
                        print(dfElt['Name'].values)
                    # print(self.linelist['Name'])
                    return 0

                line_data = self.linelist.loc[self.linelist['Name'] == line_name]
                is_in_masks  = False
                for window in masks:
                    if(window[0]< line_data.MW_Wave.values < window[1] ):
                        is_in_masks = True
                        break

                # if is_in_masks == False:
                #     print(f'{line_name}\'s wavelength ({line_data.MW_Wave:.2f} A in host galaxy rest-frame) is not included in the masks for fitting')
                #     print('This line will therfore not be included. Re-do your masks if you do want this line to be fitted')
                #     continue

                elt_index = ELEMENTS[line_data['Elt'].values[0]] - 1
                lvl_index = ION_LVL[line_data['Lvl'].values[0]] - 1

                if not self.ions_MW_fit[elt_index, lvl_index]:
                    ion = Ion(line_data['Elt'].values[0] + line_data['Lvl'].values[0], len(self.wave), 'r')
                    self.ions_MW_list.append(ion)

                self.ions_MW_fit[elt_index, lvl_index] += 1
                self.ions_MW_list[-1].list_lines.append(line_name)
                self.linelist.loc[self.linelist['Name'] == line_name, 'MW_Fit'] = 1


        else:
            print('Automatically checking MW lines falling in masks (Only a subset of strong lines are considered)')
            print()
            dfElt = self.linelist.loc[self.linelist['MW'] == 1]
            for window in masks:
                for i, row in dfElt.iterrows():
                    if(window[0]< row.MW_Wave < window[1] ):
                        print(f'{row.Name}\'s wavelength ({row.MW_Wave:.2f} A in host galaxy rest-frame) is in your masks')
                        elt_index = ELEMENTS[row.Elt] - 1
                        lvl_index = ION_LVL[row.Lvl] - 1
                        if not self.ions_MW_fit[elt_index, lvl_index]:
                            ion = Ion(row.Elt + row.Lvl, len(self.wave), 'r')
                            self.ions_MW_list.append(ion)
                        self.ions_MW_fit[elt_index, lvl_index] += 1
                        self.ions_MW_list[-1].list_lines.append(row.Name)
                        self.linelist.loc[self.linelist['Name'] == row.Name, 'MW_Fit'] = 1

        if(len(self.ions_MW_list) > 0):
             self.MW_fit = 1

        return 1

    def print_MW_lines_to_fit(self, z) -> None:
        print('------------ List of MW lines included in the fit --------------')
        for i, element in enumerate(ELEMENTS.keys()):
            n_ions_fit = np.sum(self.ions_MW_fit[i])
            if n_ions_fit > 0:
                print(f' {element} with {n_ions_fit:.0f} line(s) included:')
                for j, ion_level in enumerate(ION_LVL.keys()):
                    if self.ions_MW_fit[i][j] > 0:
                        print(f'\t -- {element} {ion_level} ;')
                        temp = self.linelist[(self.linelist['Elt'] == element) &
                                                   (self.linelist['Lvl'] == ion_level)&
                                                self.linelist['MW_Fit'] == 1]
                        print(temp[['Name', 'MW_Wave']].to_string(index=False))
                        print('-----------')


    def define_masks(self, wave, flux, masks = None, interactive = 0):
        if(interactive == 1):
            masks = []
            temp = self.linelist[self.linelist['Fit'] == 1]
            temp = temp.sort_values('Wave')
            temp = temp.reset_index(drop=True)
            for i in range(len(temp)):
                fig,ax  = plt.subplots(figsize = (8,4))
                plt.plot(wave, flux, c= 'k',drawstyle='steps-mid')
                plt.xlim(temp.loc[i]['Wave']-4, temp.loc[i]['Wave']+4)
                plt.ylim(0,2)
                plt.title(f"Selecting windows for {temp.loc[i]['Name']} (x-grid is 0.2A)", fontsize = 15)
                ax.tick_params(axis="both", direction="in", which="both")
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.xaxis.set_minor_locator(MultipleLocator(0.2))
                plt.grid(which = 'both', axis = 'x', linestyle = '--')
                for (xmin, xmax) in masks:
                    if(temp.loc[i]['Wave']-5 < xmin < temp.loc[i]['Wave']+5 or
                     temp.loc[i]['Wave']-5 < xmax < temp.loc[i]['Wave']+5):
                        plt.plot(wave[(wave>xmin) & (wave<xmax)],
                                 flux[(wave>xmin) & (wave<xmax)],
                                 c= 'mediumseagreen',drawstyle='steps-mid')
                plt.show()
                user_input = ''

                while(True):
                    user_input = input('Write new interval to add as "xmin, xmax". Write done to go to next wavelength window: ')
                    if(user_input == 'done'):
                        break
                    try:
                        x_range = user_input.split(',')
                        xmin = float(x_range[0].strip())
                        xmax = float(x_range[1].strip())
                        masks.append((xmin, xmax))

                    except (ValueError, IndexError):
                        print("Invalid input: '{}', retry with format [xmin, xmax]".format(user_input))
                fig,ax  = plt.subplots(figsize = (8,4))
                plt.plot(wave, flux, c= 'k',drawstyle='steps-mid')
                for (xmin, xmax) in masks:
                    if(temp.loc[i]['Wave']-5 < xmin < temp.loc[i]['Wave']+5 or
                     temp.loc[i]['Wave']-5 < xmax < temp.loc[i]['Wave']+5):
                        plt.plot(wave[(wave>xmin) & (wave<xmax)],
                                 flux[(wave>xmin) & (wave<xmax)],
                                 c= 'mediumseagreen',drawstyle='steps-mid')
                plt.xlim(temp.loc[i]['Wave']-4, temp.loc[i]['Wave']+4)
                plt.ylim(0,2)
                ax.tick_params(axis="both", direction="in", which="both")
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.xaxis.set_minor_locator(MultipleLocator(0.2))
                plt.grid(which = 'both', axis = 'x', linestyle = '--')
                plt.title(f"Spectral windows (green) for {temp.loc[i]['Name']}", fontsize = 15)
                plt.show()

            print('You have selected the following masks', masks)

        elif(masks == None):
            masks = []
            temp = self.linelist.loc[self.linelist['Fit'] == 1].sort_values('Wave').reset_index(drop=True)
            for i in range(len(temp)):
                masks.append([temp.loc[i]['Wave']-3, temp.loc[i]['Wave']+1.5])
        else:
            print('The user is providing his own masks (good job user!)')


        return masks


    def create_model_1comp(self, lmfit_parameters=None, x=None):
        x = x if x is not None else self.wave
        lmfit_parameters = lmfit_parameters if lmfit_parameters is not None else self.init_parameters
        model = np.ones(len(x))
        temp  = self.linelist[self.linelist['Fit'] == 1].reset_index(drop=True)

        for i, row in temp.iterrows():
            ion = row['Elt'] + row['Lvl']
            model *= (lmfit_parameters['fcov'+ion]*self.voigt_profile(x, lmfit_parameters["v"+ion],
                                                                     lmfit_parameters["b"+ion],
                                                                     lmfit_parameters["logn"+ion],
                                                                     row["Wave"],
                                                                     row["f"],
                                                                     row["A"])
                     + (1 - lmfit_parameters['fcov'+ion]))


        if(self.MW_fit):
            temp  = self.linelist[self.linelist['MW_Fit'] == 1].reset_index(drop=True)

            for i, row in temp.iterrows():
                ion = row['Elt'] + row['Lvl']
                model *= (lmfit_parameters['fcovMW'+ion]*self.voigt_profile(x, lmfit_parameters["vMW"+ion],
                                                                         lmfit_parameters["bMW"+ion],
                                                                         lmfit_parameters["lognMW"+ion],
                                                                         row["MW_Wave"],
                                                                         row["f"],
                                                                         row["A"])
                         + (1 - lmfit_parameters['fcovMW'+ion]))

        return model


    def model_1_lmfit(self, lmfit_parameters, x, y, yerr):
        model =self.create_model_1comp(lmfit_parameters,x)
        return (y-model)/yerr

    def initialize_parameters(self):
        params = Parameters()

        for ion in self.ions_list:
            params.add("v"+ion.name, value = ion.v[0], min =-500, max = 200)
            params.add("b"+ion.name, value = ion.b[0], min =0., max = 300.)
            if ion.name == 'HI':
                params.add("logn"+ion.name, value = ion.logn[0], min =12.0, max = 22.)
            else:
                params.add("logn"+ion.name, value = ion.logn[0], min =11.0, max = 18.)

            # params.add("logn"+ion.name, value = ion.logn[0], min =10.0, max = 23.)
            params.add("fcov"+ion.name, value = ion.fcov[0], min =.0, max = 1.)

            print('%s initial parameters with the following values:'%ion.name)
            print('\t Velocity (v%s) - value %0.f - Bounds [%.0f, %.0f]'%(ion.name, params['v'+ion.name].value, params['v'+ion.name].min, params['v'+ion.name].max))
            print('\t Doppler parameter (b%s) - value %0.f - Bounds [%.0f, %.0f]'%(ion.name,params['b'+ion.name].value, params['b'+ion.name].min, params['b'+ion.name].max))
            print('\t log10 Column density (logn%s) - value %0.f - Bounds [%.0f, %.0f]'%(ion.name,params['logn'+ion.name].value, params['logn'+ion.name].min, params['logn'+ion.name].max))
            print('\t Covering fraction: (fcov%s) - value %0.f - Bounds [%.0f, %.0f]'%(ion.name,params['fcov'+ion.name].value, params['fcov'+ion.name].min, params['fcov'+ion.name].max))
            print()

        self.init_parameters = params
        return


    def initialize_MW_parameters(self):
        params = self.init_parameters
        for ion in self.ions_MW_list:
            params.add("vMW"+ion.name, value = 0, min =-40, max = 20)
            params.add("bMW"+ion.name, value = 30, min =0., max = 50., vary = False)
            if ion.name == 'HI':
                params.add("lognMW"+ion.name, value = 19, min =18.0, max = 23.)
            else:
                params.add("lognMW"+ion.name, value = 14, min =13.0, max = 15.5)

            params.add("fcovMW"+ion.name, value = 1, min =.0, max = 1., vary = False)

            print('%s initial parameters with the following values:'%ion.name)
            print('\t Velocity (vMW%s) - value %0.f - Bounds [%.0f, %.0f]'%(ion.name, params['vMW'+ion.name].value, params['vMW'+ion.name].min, params['vMW'+ion.name].max))
            print('\t Doppler parameter (bMW%s) - value %0.f , fixed'%(ion.name,params['bMW'+ion.name].value))
            print('\t log10 Column density (lognMW%s) - value %0.f - Bounds [%.0f, %.0f]'%(ion.name,params['lognMW'+ion.name].value, params['lognMW'+ion.name].min, params['lognMW'+ion.name].max))
            print('\t Covering fraction: (fcovMW%s) fixed to 1'%(ion.name))
            print()

        # self.init_MW_parameters = params
        return

    def update_parameter(self, name, value = None, minV = None, maxV = None, expr = None, vary = None):
        # try:
        if value is not None:
            self.init_parameters[name].value = value
        if minV is not None:
            self.init_parameters[name].min = minV
        if maxV is not None:
            self.init_parameters[name].max = maxV
        if expr is not None:
            self.init_parameters[name].expr = expr
        if vary is not None:
            self.init_parameters[name].vary = vary

        if(self.init_parameters[name].expr is None):
            print('Parameters updated for %s: value %0.f - Bounds [%.0f, %.0f]'%(name,self.init_parameters[name].value,
                                                                                      self.init_parameters[name].min,
                                                                                      self.init_parameters[name].max))
        else:
            print('Parameters updated for %s: new expression %s'%(name,self.init_parameters[name].expr))

    def update_MW_parameter(self, name, value = None, minV = None, maxV = None, expr = None, vary = None):
        # try:
        if value is not None:
            self.init_parameters[name].value = value
        if minV is not None:
            self.init_parameters[name].min = minV
        if maxV is not None:
            self.init_parameters[name].max = maxV
        if expr is not None:
            self.init_parameters[name].expr = expr

        if(self.init_parameters[name].expr is None):
            print('Parameters updated for %s: value %0.f - Bounds [%.0f, %.0f]'%(name,self.init_parameters[name].value,
                                                                                      self.init_parameters[name].min,
                                                                                      self.init_parameters[name].max))
        else:
            print('Parameters updated for %s: new expression %s'%(name,self.init_parameters[name].expr))

    def run_fit(self, x, y, yerr = None, params=None, verbose=False, method = 'ampgo'):
        """
        Perform a fit of the model to the data and update the ion parameters.

        Parameters:
        -----------
        params : lmfit.Parameters object or None
            The parameters to be used in the fit. If None, the parameters
            previously set in the object will be used.
        verbose : bool
            Whether to print fitting and ion parameters information and plot the final fit.

        Returns:
        --------
        final_parameters : lmfit.Parameters object
            The final parameters obtained after the fit.

        """
        # If no parameters are passed, use the ones previously set in the object
        params = params if params is not None else self.init_parameters
        yerr = yerr if yerr is not None else np.ones(len(x))

        if(method != 'ampgo'):
            print('If you do not use a global optimization solver, it is advised to perform the fit several times and use median & IQ')
            # Perform the fit using the ampgo solver and update the ion parameters
        print('Default absorption fitting approach with %s solver, 1 iteration only'%method)
        print('Launching fit')
        print()

        start_time = time.time()
        self.fit = np.ones(len(self.wave))
        self.final_parameters = minimize(self.model_1_lmfit, params, args=(x,y,yerr), method=method, nan_policy='omit')
        print('Fit done! Time elapsed %.2f s'% (time.time() - start_time))
        print()
        # Print fitting and ion parameters information and plot the final fit if verbose
        if verbose:
            print('Fitting report from lmfit:')
            print(fit_report(self.final_parameters))
            print()

        for ion in self.ions_list:
            spectrum = np.ones(len(self.wave))
            ion.update_parameters(self.final_parameters)
            for row in self.linelist[(self.linelist['Fit']==1) & (self.linelist['Elt']+self.linelist['Lvl'] == ion.name)].itertuples(index=False):
                spectrum *= (self.final_parameters.params['fcov'+ion.name].value *
                             self.voigt_profile(self.wave, self.final_parameters.params["v"+ion.name].value,
                                                self.final_parameters.params["b"+ion.name].value,
                                                self.final_parameters.params["logn"+ion.name].value,
                                                row.Wave, row.f, row.A) +
                             (1 - self.final_parameters.params['fcov'+ion.name].value))
            self.fit *= spectrum
            ion.update_spectrum(spectrum)

        if(self.MW_fit):
            for ion in self.ions_MW_list:
                spectrum = np.ones(len(self.wave))
                ion.update_MW_parameters(self.final_parameters)
                for row in self.linelist[(self.linelist['MW_Fit']==1) & (self.linelist['Elt']+self.linelist['Lvl'] == ion.name)].itertuples(index=False):
                    spectrum *= (self.final_parameters.params['fcovMW'+ion.name].value *
                                 self.voigt_profile(self.wave, self.final_parameters.params["vMW"+ion.name].value,
                                                    self.final_parameters.params["bMW"+ion.name].value,
                                                    self.final_parameters.params["lognMW"+ion.name].value,
                                                    row.MW_Wave, row.f, row.A) +
                                 (1 - self.final_parameters.params['fcovMW'+ion.name].value))
                self.fit *= spectrum
                ion.update_spectrum(spectrum)

            # elif(method == 'leastsq'):
            #     nit = 20
            #     for it in range(nit):
            #         newy =
            #         self.fit = np.ones(len(self.wave))
            #         self.final_parameters = minimize(self.model_1_lmfit, params, args=(x,y,yerr), method=method, nan_policy='omit')
            #     all_params = np.zeros((len(self.ions_list), 20, 4))
            #     i = 0
            #     for ion in self.ions_list:
            #         all_params[i,]

            return self.final_parameters


    #
