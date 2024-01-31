#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 2022

@author: xiaowei
"""

import os
import numpy as np
import pickle
from astropy.table import Column, Table, join, vstack, hstack
from astropy.io import fits, ascii
import os.path
import subprocess

from normalize_spectra_smhr import LoadAndNormalizeData

# -------------------------------------------------------------------------------
# load catalogs
# -------------------------------------------------------------------------------

print("opening Gaia eDR3 and APOGEE cross match catalogue. ")

#tbl1 = Table.read('data/allStar-r12-gaiadr2.fits', 1)
# tbl1 = Table.read('data/allStar-r12-gaiaedr3-xmatch.fits', 1)
tbl1 = Table.read('data/spec_plx_a17cand_gaiadr3_allwise_match.fits', 1)
# tbl1.rename_column('apogee_id', 'APOGEE_ID')
#tbl2 = Table.read('data/allStar-r12-l33.fits', 1)
col_APOGEE = ['APOGEE_ID','LOGG','J','H','K','J_ERR','H_ERR','K_ERR','EXTRATARG','GAIAEDR3_SOURCE_ID','TELESCOPE','FIELD']
tbl2 = Table.read('../../Fa2020_MW_substructure/data/allStarLite-dr17-synspec_rev1.fits', 1)
# Only keep the columns we need for later
tbl2 = tbl2[col_APOGEE]

print(len(tbl1),len(tbl2))

# training_lab = join(tbl1, tbl2, keys='APOGEE_ID') # some issue with APOGEE_ID matching
training_lab = hstack([tbl1, tbl2]) # assume the two table are in the same order, wihch they should



print("Gaia eDR3 and APOGEE DR17 cross match catalogue: {} entries. ".format(len(training_lab))) 

# Calculate and rename some additional columns
fn_tmp = ['']*len(training_lab)
for i in range(len(training_lab)):
    if training_lab['TELESCOPE'][i] == 'lco25m':
        fn_tmp[i] = f"asStar-dr17-{str(training_lab['APOGEE_ID'][i]).strip()}.fits"
    else:
        fn_tmp[i] = f"apStar-dr17-{str(training_lab['APOGEE_ID'][i]).strip()}.fits"
training_lab.add_column(fn_tmp, name='FILE')

# for Gaia archive match! add # in text file before header!
# ascii.write(tbl1['source_id', 'APOGEE_ID'], 'data/apogeedr16_edr3_sources.txt')

# -------------------------------------------------------------------------------
# cut in logg... take only RGB stars!
# -------------------------------------------------------------------------------

parent_logg_cut = 2.2
cut = np.logical_and(training_lab['LOGG'] <= parent_logg_cut, training_lab['LOGG'] > 0.)
training_lab = training_lab[cut]              
print('logg <= {0} cut: {1}'.format(parent_logg_cut, len(training_lab)))

# -------------------------------------------------------------------------------
# cut in Q_K (where training_labels['K'] was missing! (i.e. 99)) # Note from XO: bad values changed from -999 to 99 in DR17
# ------------------------------------------------------------------------------- 

cut = np.where(training_lab['K'] < 90)
training_lab = training_lab[cut]              
print('remove missing K: {}'.format(len(training_lab)))

# -------------------------------------------------------------------------------
# remove missing data
# ------------------------------------------------------------------------------- 

cut = np.where(np.isfinite(training_lab['bp_rp']))
training_lab = training_lab[cut]
print('remove bad bp-rp: {}'.format(len(training_lab)))

cut = np.where(np.isfinite(training_lab['parallax']))
training_lab = training_lab[cut]
print('remove bad plx: {}'.format(len(training_lab)))


# -------------------------------------------------------------------------------
# add WISE catalog -- changed to using prematched columns
# -------------------------------------------------------------------------------

print('match to WISE...')
#hdu = fits.open('data/gaia_apogeedr16_wise.fits')
# hdu = fits.open('data/wise_apogee_dr16_edr3.fits')
# wise_data = Table.read('data/spec_plx_a17cand_allwise_match.fits',1)
# wise_data.rename_column('apogee_id','APOGEE_ID')
# Remove the missing source_ids (i.e. no match)
# wise_data = wise_data[~wise_data['source_id'].mask]
# training_labels = join(training_lab, wise_data, keys='source_id')

training_labels = training_lab[~training_lab['original_ext_source_id'].mask]

print('matched: {}'.format(len(training_labels))) 

# -------------------------------------------------------------------------------
# check for existing WISE colors
# -------------------------------------------------------------------------------

cut = np.isfinite(training_labels['w2mpro'])
training_labels = training_labels[cut] 
cut = np.isfinite(training_labels['w1mpro'])
training_labels = training_labels[cut] 
print('remove missing W1 and W2: {}'.format(len(training_labels))) 

# -------------------------------------------------------------------------------
# check for potential mismatch in WISE
# -------------------------------------------------------------------------------

cut = (training_labels['number_of_neighbours'] == 1) & (training_labels['number_of_mates'] == 0)
training_labels = training_labels[cut]
print('remove potential mismatch in WISE: {}'.format(len(training_labels))) 

# -------------------------------------------------------------------------------
# take only unique entries!
# -------------------------------------------------------------------------------

# might be better to make a quality cut here!
badbits = 2**4
cut = np.bitwise_and(training_labels['EXTRATARG'],badbits) == 0
training_labels = training_labels[cut]
# uni, uni_idx = np.unique(training_labels['APOGEE_ID'], return_index = True)
# training_labels = training_labels[uni_idx]
print('remove duplicats: {}'.format(len(training_labels)))

# -------------------------------------------------------------------------------
# remove variable stars (at least those marked by Gaia)
# ------------------------------------------------------------------------------- 

#cut = training_labels['phot_variable_flag'] != 'VARIABLE'
#training_labels = training_labels[cut]              
#print('remove variable stars: {}'.format(len(training_labels)))

#Table.write(training_labels['source_id', 'RA', 'DEC'], 'data/gaiadr2_apogeedr15_IDs.txt', format = 'ascii', overwrite = True)
    
        
# -------------------------------------------------------------------------------
# calculate MAD
# -------------------------------------------------------------------------------

# remove missing files (used after we had at least run the downloading script below once)
# mad = np.zeros((len(training_labels)))
# snr = np.zeros((len(training_labels)))
# found = np.ones((len(training_labels)), dtype=bool)
# destination = './data/spectra/'
# for i in range(len(training_labels['FILE'])):
#     entry = destination + (training_labels['FILE'][i]).strip()
#     try:
#         hdulist = fits.open(entry) 
#         data = hdulist[1].data[0]
#         mad[i] = np.median(np.abs(data[:-1]-data[1:]))
#         snr[i] = training_labels['SNR'][i]
#         print(i, 'found!')
#     except:
#         print(i, training_labels['FILE'][i], training_labels['FIELD'][i], training_labels['TELESCOPE'][i])
#         found[i] = False 
  
# t = Table([mad, snr, found], names = ('MAD', 'SNR', 'found'))
# ascii.write(t, 'mad_snr.txt', overwrite = True)
        
# -------------------------------------------------------------------------------
# download spectra
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# create a list of file names for download
# -------------------------------------------------------------------------------

test_len = 100

fspeclist = open(f"speclist.txt", 'w')
for i, (fn, tel, field) in enumerate(zip(training_labels['FILE'][:test_len], training_labels['TELESCOPE'][:test_len], training_labels['FIELD'][:test_len])):
    file = str(tel).strip() + '/' + str(field).strip() + '/' + str(fn).strip()
    fspeclist.writelines(f"{file}\n")
fspeclist.close()

# Temporary debugging
# quit()

delete0 = 'find ./data/spectra/ -size 0c -delete' # Delete empty files in the data directory
subprocess.call(delete0, shell = True)

# Download using wget
# cmd = 'wget -nv -r -nH --cut-dirs=9 -i speclist.txt -P ./data/spectra/ -B https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/'
# print(cmd)
# subprocess.call(cmd, shell = True)
# subprocess.call(delete0, shell = True)


        
# remove missing files 
found = np.ones_like(np.arange(len(training_labels[:test_len])), dtype=bool)
destination = './data/spectra/'
for i in range(len(training_labels['FILE'][:test_len])):
    entry = destination + (training_labels['FILE'][i]).strip()
    try:
        hdulist = fits.open(entry)
    except:
        print(entry + " not found or corrupted; deleting!")
        cmd = 'rm -vf ' + entry 
        subprocess.call(cmd, shell = True)
        print(i, training_labels['FILE'][i], training_labels['FIELD'][i], training_labels['TELESCOPE'][i])
        found[i] = False   

training_labels = training_labels[:test_len][found]
print('spectra found for: {}'.format(len(training_labels)))


# -------------------------------------------------------------------------------
# save training labels
# -------------------------------------------------------------------------------

print('save labels...')
training_labels.write('data/training_labels_parent_apogeedr17_edr3_test.fits', overwrite = True)

# -------------------------------------------------------------------------------
# normalize spectra
# -------------------------------------------------------------------------------

print('load and normalize spectra...')
file_name = 'data/all_flux_norm_parent_apogeedr17_edr3_test.fits'
data_norm, continuum, found = LoadAndNormalizeData(training_labels['FILE'], 'data/spectra', file_name)
print('found: {}'.format(np.sum(found)))



# print('save labels...')
# Table.write(training_labels, 'data/training_labels_parent_apogeedr16_edr3_2.fits', format = 'fits', overwrite = True)
    

## remove entries from training labels, where no spectrum was found (for whatever reason...)!
#f = open('data/no_data_parent.pickle', 'rb') 
#no_dat = pickle.load(f) 
#f.close()
#training_labels = training_labels[no_dat]  
#print('remove stars with missing spectra: {}'.format(len(training_labels)))
                
# -------------------------------------------------------------------------------



