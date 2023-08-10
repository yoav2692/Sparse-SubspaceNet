"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: Y. Amiel
    Created: 09/08/28
    Edited: 09/08/28

    Purpose
    --------
    get input from terminal to use multiple machines

"""
# Imports

from experiments_handler import *

if __name__ == "__main__":
    num_sources = int(input())
    ExperimentSetup.experiment_sweep(num_sources)


