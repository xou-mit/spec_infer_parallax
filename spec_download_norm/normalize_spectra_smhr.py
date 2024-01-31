#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 2022

@author: xo
"""

import numpy as np
from alexmods.specutils import Spectrum1D
import os.path
from astropy.io import fits
import traceback

_DEFAULT_NORMKWARGS = {"function":"spline", "high_sigma_clip": 0.1, "include": None,
                    "knot_spacing": 100, "low_sigma_clip": 1.0, "max_iterations": 5, "order": 3, "scale": 1.0,
                    "blue_trim": 0, "red_trim": 0}

_DEFAULT_BADBITMASK = 61951 # '1111000111111111'

def LoadAndNormalizeData(file_name, in_dir, out_f, badbit=_DEFAULT_BADBITMASK, norm_kwargs=_DEFAULT_NORMKWARGS):
    '''
    load APOGEE spectra in .fits, normalize, and return the normalized spectra as well as the continuum
    Input:
    file_name: list of file names of the spectra
    in_dir: input directory name, where unnormalized spectra are read in
    out_f: output file name, where normalized spectra are stored
    
    Returns:
    spec_norm: normalized spectra
    continuum: normalized continuum
    '''
    all_norm_disp = np.zeros((len(file_name), 8575))
    all_norm_flux = np.zeros((len(file_name), 8575))
    all_norm_sig = np.zeros((len(file_name), 8575))
    all_cont = np.zeros((len(file_name), 8575))
    LARGE  = 3.0 # magic LARGE sigma value
    i=0
    no_data = []
    no_data_i = np.ones((len(file_name),), dtype = bool)
    for entry in file_name:
        print(i)
        
        try:
            # Extract data and check the dimension
            hdulist = fits.open(os.path.join(in_dir, entry.strip()))
            if len(hdulist[1].data) < 8575: 
                flux = hdulist[1].data[0]
                sigma = hdulist[2].data[0]
                mask = hdulist[3].data[0] 
            else:
                flux = hdulist[1].data
                sigma = hdulist[2].data
                mask = hdulist[3].data[0]
                print(f"Spectra {entry} has only one visit in APOGEE!")
            # Compute dispersion
            header = hdulist[1].header
            start_wl = header['CRVAL1']
            diff_wl = header['CDELT1']
            val = diff_wl * (len(flux)) + start_wl
            wl_full_log = np.arange(start_wl, val, diff_wl)
            wl_full = np.array([10**aval for aval in wl_full_log])
            # Mask bad pixels but keep the length the same
            # Default keeping pixels only flagged with low/medium/high persistence
            # Bit 9, 10, and 11
            # Additional test of NaN needed to identify gap pixels; they all have bitmask = 0
            gd_pix = (np.bitwise_and(mask, badbit) == 0) & (np.isnan(flux) == False)
            flux[~gd_pix] = -1 # Set bad pixels to some negative flux so SMHR ignores them
            Nwl_masked = len(wl_full)
            spec = Spectrum1D(wl_full,flux,1/sigma**2)
            norm, cont, left, right = spec.fit_continuum(**norm_kwargs,full_output=True)
            print(left, right)
            # Might need to stuff the bad pixels and snipped edges back in...
            # First put back the snipped edges during SMHR normalization
            disp_norm = np.concatenate((spec.dispersion[:left],norm.dispersion,spec.dispersion[right:]))
            flux_norm = np.concatenate((np.array([1.0]*left),norm.flux,np.array([1.0]*(Nwl_masked-right))))
            ivar_norm = np.concatenate((np.array([LARGE]*left),norm.ivar,np.array([LARGE]*(Nwl_masked-right))))
            # Then put back the bad pixels masked before normalization; Check with Christina if these should be some specific fill-in values
            # make arrays will fill-in values and then put in the valid values: setting norm to 1 and sigma to 3 for bad pixels
            disp_norm_full, flux_norm_full, ivar_norm_full = disp_norm, np.ones(8575), np.ones(8575)*LARGE
            flux_norm_full[gd_pix] = flux_norm[gd_pix]
            ivar_norm_full[gd_pix] = ivar_norm[gd_pix]
            sig_norm_full = np.sqrt(1/ivar_norm_full)
            cont_full = cont
            # search through the normalized wavelength range and 
            # Store the result
            # print(f"output normalized dispersion length is: {len(mask)} --> {len(wl_full)} --> {len(norm.dispersion)} --> \
            #     {len(disp_norm)} --> {len(disp_norm_full)}")
            # print(f"output normalized wavelength length is: {len(mask)} --> {len(flux)} --> {len(norm.flux)} --> \
            #     {len(flux_norm)} --> {len(flux_norm_full)}")
            # print(f"output normalized sigma length is: {len(mask)} --> {len(sigma)} --> {len(norm.ivar)} --> \
            #     {len(ivar_norm)} --> {len(sig_norm_full)}")
            all_norm_disp[i] = disp_norm_full
            all_norm_flux[i] = flux_norm_full
            all_norm_sig[i] = sig_norm_full
            all_cont[i] = cont_full
        except Exception as err:
            no_data_i[i] = False
            no_data.append(entry)
            print(traceback.format_exc())
        i += 1
    
    # data_norm = np.array([all_norm_disp[no_data_i, :], all_norm_flux[no_data_i, :], all_norm_sig[no_data_i, :], all_cont[no_data_i, :]])
    data_norm = np.array([all_norm_flux[no_data_i, :], all_norm_sig[no_data_i, :], all_cont[no_data_i, :]])
    # print(data_norm.shape, all_norm_flux.shape, all_norm_sig.shape)
    # data_tp = data_norm.T
    # print(data_tp.shape)
    all_cont = all_cont[no_data_i, :]
    
    fits.writeto(out_f, data_norm.T, overwrite = True)
    
    return data_norm, all_cont, no_data_i

if __name__=="__main__":
    # Test the normalization scheme

    # Create a list of file names
    in_spec_f = ['apStar-dr17-2M00000002+7417074.fits','asStar-dr17-2M00000035-7323394.fits',
    'asStar-dr17-2M03592391+1506513.fits','asStar-dr17-2M04024943+1626023.fits',
    'asStar-dr17-2M03585730+2123454.fits','apStar-dr17-2M19331341-1923240.fits',
    'apStar-dr17-2M00001719+6221324.fits','apStar-dr17-2M00001653+5540107.fits']

    # Call the function
    norm_spec, continuum, no_data_i = LoadAndNormalizeData(in_spec_f,'data/spectra','test_norm_data.fits')