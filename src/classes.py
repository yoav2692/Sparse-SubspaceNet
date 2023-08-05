"""Subspace-Net 
Details
----------
Name: classes.py
Authors: Y. Amiel
Created: 05/08/23
Edited: 05/08/23

Purpose:
--------
This script defines all classes of the simulation and defines their defaults.
"""

# Imports
import numpy as np
from enum import Enum

class Signal_nature(Enum):
    DEFAULT = "non-coherent"
    coherent = "coherent"
    non_coherent = "non-coherent"

class Signal_type(Enum):
    DEFAULT    = "narrowband"
    narrowband = "narrowband"
    broadband = "broadband"

class Model_type(Enum):
    DEFAULT    = "SubspaceNet"
    SubspaceNet = "SubspaceNet"
    MatrixCompletion = "MatrixCompletion"
    DeepCNN = "DeepCNN"

class matrix_completion_method(Enum):
    spatial_stationary = "spatial_stationary"
    low_rank = "low_rank"

class detection_method(Enum):
    DEFAULT    = "esprit"
    esprit = "esprit"
    root_music = "root_music"

class Criterion(Enum):
    DEFAULT    = "RMSPE"
    RMSPE = "RMSPE"
    BCE = "BCE"

class Loss_method(Enum):
    DEFAULT    = "no_permute"
    no_permute = "no_permute"
    full_permute = "full_permute"
    # closest_no_repetiotions = "closest_no_repetiotions"

class Optimizer(Enum):
    DEFAULT = "Adam"
    Adam = "Adam"
    SGD = "SGD"
    SGD_Momentum = "SGD Momentum"
    