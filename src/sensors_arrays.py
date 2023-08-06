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
import re
from src.classes import *
import torch
import random
import scipy

MRA_DIFFS = {'MRA-4' : [1,3,2] , 
'MRA-81' : [8,10,1,3,2,7,8],
'MRA-82': [1,3,6,6,2,3,2]}

class SensorsArrayGenerator():
    def __init__(self,sensors_array_form):
        self.sensors_array_form = sensors_array_form
        self.generating_array = None
    
    def generate(self,sensors_array_form):
        if "-complementary" in sensors_array_form:
            locs , last_sensor_loc =  self.generate(sensors_array_form.rsplit("-complementary")[0])
            comp_locs = np.array([x for x in range(last_sensor_loc) if x not in locs ])
            return comp_locs , last_sensor_loc
        else:
            if self.sensors_array_form.startswith("MRA"):
                match = re.search('MRA-(\d+)',sensors_array_form)
                key = "MRA-" + match.group(1)
                if key not in MRA_DIFFS.keys():
                    raise Exception(f"{match.group(1)} is not suppurted")
                self.generating_array = MRA_DIFFS[key]
                locs = np.insert( np.cumsum(self.generating_array),0,0)
                last_sensor_loc = locs[-1] + 1
                return locs , last_sensor_loc
            elif self.sensors_array_form.startswith("ULA"):
                match = re.search('ULA-(\d+)',sensors_array_form)
                num_sensors = int(match.group(1))
                return np.array(range(num_sensors)) , num_sensors
            else:
                raise Exception(f"{self.sensors_array_form} is not yet supported")

class SensorsArray():
    def __init__(self,sensors_array_form:str , missing_sensors_handle_method: str = Missing_senors_handle_method.DEFAULT.value ):
        self.sensors_array_form = sensors_array_form
        self.sparsity_type = self.sensors_array_form.rsplit('-')[0]
        self.generator = SensorsArrayGenerator(self.sensors_array_form)
        self.missing_sensors_handle_method = missing_sensors_handle_method
        self.locs , self.last_sensor_loc = self.generator.generate(self.sensors_array_form)
        self.num_sensors = len(self.locs)
        if "-virtualExtention" in sensors_array_form:
            match = re.search('-virtualExtention-(\d+)',sensors_array_form)
            self.set_last_sensor_loc(int(match.group(1)))
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
        if len(holes[0]) == 0: # no holes check
            return self.last_sensor_loc
        return holes[0]

    def set_last_sensor_loc(self,last_sensor_loc):
        # to support experimental antenna formation like: xx--x-x------
        self.last_sensor_loc = last_sensor_loc 

if __name__ == "__main__":
    pass
    # y = SensorsArray("ULA-4")
    # x = SensorsArray("MRA-4-complementary")
    
