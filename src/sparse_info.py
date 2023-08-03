"""Subspace-Net 
Details
----------
Name: utils.py
Authors: Y. Amiel
Created: 01/08/23
Edited: 01/08/23

Purpose:
--------
This script defines sparse arrays information:
    
"""

# Imports
import numpy as np
import torch
import random
import scipy

# Functions
def calc_all_diffs(array , negative_handle = 'pos_only'):
    diff_array = np.zeros((array[-1] + 1,1))
    diff_array[0] = len(array)
    for first_ind , first_obj in enumerate(array):
        for second_obj in array[first_ind+1:]:
            if negative_handle == 'pos_only':
                diff_array[second_obj-first_obj] += 1
    return diff_array
# Constants

supported_arrays = ['MRA-4']
'''
# Dictionaries
mra_difs = {'MRA-4' : [1,3,2] , 
'MRA-8' : [8,10,1,3,2,7,8],
'MRA-8_r': [1,3,6,6,2,3,2]}
mra_locs = mra_difs
mra_virtual_ants = mra_difs
for i,k in enumerate(mra_difs):
    mra_locs[k] = np.insert( np.cumsum(mra_difs[k]),0,0)
    mra_locs_temp = mra_locs[k]
    mra_virtual_ants[k] = np.where(calc_all_diffs(mra_locs_temp) == 0)[0]
    if len(mra_virtual_ants[k]) == 0:
        mra_virtual_ants[k] = mra_locs[k][-1]+1 

MRA_DIFFS = mra_difs
MRA_LOCS = mra_locs
MRA_VIRTUAL_ANTS = mra_virtual_ants
'''
MRA_DIFFS = {}
MRA_DIFFS['MRA-4'] = [1,3,2]
MRA_LOCS = {}
MRA_LOCS['MRA-4'] = [0,1,4,6]
MRA_LOCS['MRA-4-complementary'] = [2,3,5]
MRA_VIRTUAL_ANTS = {}
MRA_VIRTUAL_ANTS['MRA-4'] = 7
MRA_VIRTUAL_ANTS['MRA-4-complementary'] = 7

if __name__ == "__main__":
    pass
