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

MRA_DIFFS = {'MRA-4' : [1,3,2] , 
'MRA-8' : [8,10,1,3,2,7,8],
'MRA-8_r': [1,3,6,6,2,3,2]}
MRA_LOCS = {}
MRA_LOCS['MRA-4'] = [0,1,4,6]
MRA_LOCS['MRA-4-complementary'] = [2,3,5]
MRA_VIRTUAL_ANTS = {}
MRA_VIRTUAL_ANTS['MRA-4'] = 7
MRA_VIRTUAL_ANTS['MRA-4-complementary'] = 7

class SensorsArrayGenerator():
    def __init__(self,sparse_form):
        self.sparse_form = sparse_form
        self.generating_array = None
    
    def generate(self,sparse_form):
        if "-complementary" in sparse_form:
            locs , last_sensor_loc =  self.generate(sparse_form.rsplit("-complementary")[0])
            comp_locs = np.array([x for x in range(last_sensor_loc) if x not in locs ])
            return comp_locs , last_sensor_loc
        else:
            if self.sparse_form.startswith("MRA"):
                self.generating_array = MRA_DIFFS[sparse_form]
                locs = np.insert( np.cumsum(self.generating_array),0,0)
                last_sensor_loc = locs[-1] + 1
                return locs , last_sensor_loc
            elif self.sparse_form.startswith("ULA"):
                num_sensors = int(self.sparse_form.rsplit("ULA-")[1])
                return np.array(range(num_sensors)) , num_sensors
            else:
                raise Exception(f"{self.sparse_form} is not yet supported")

class SensorsArray():
    def __init__(self,sparse_form:str):
        self.sparse_form = sparse_form
        self.sparsity_type = self.sparse_form.rsplit('-')[0]
        self.generator = SensorsArrayGenerator(self.sparse_form)
        self.locs , self.last_sensor_loc = self.generator.generate(self.sparse_form)
        self.num_sensors = len(self.locs)
        self.num_virtual_sensors = self.set_virtual_sensors()

    # Functions
    def calc_all_diffs( self, negative_handle = 'pos_only'):
        diff_array = np.zeros((self.last_sensor_loc,1))
        diff_array[0] = self.num_sensors
        for first_ind , first_obj in enumerate(self.locs):
            for second_obj in self.locs[first_ind+1:]:
                if negative_handle == 'pos_only':
                    diff_array[second_obj-first_obj] += 1
        return diff_array

    def set_virtual_sensors(self):
        holes = np.where(self.calc_all_diffs() == 0)
        if len(holes[0]) == 0:
            return self.last_sensor_loc
        return holes[0]

    def set_last_sensor_loc(self,last_sensor_loc):
        self.last_sensor_loc = last_sensor_loc

if __name__ == "__main__":
    y = SensorsArray("ULA-4")
    x = SensorsArray("MRA-4-complementary")
    
