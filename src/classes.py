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
    DEFAULT    = "NarrowBand"
    narrowband = "NarrowBand"
    broadband = "BroadBand"

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
    no_permute_periodic = "no_permute_periodic" # temp for POC
    full_permute_periodic = "full_permute_periodic"
    # closest_with_repetiotions = "closest_with_repetiotions"
    # stable_match = "stable_match"

class Optimizer(Enum):
    DEFAULT = "Adam"
    Adam = "Adam"
    SGD = "SGD"
    SGD_Momentum = "SGD Momentum"
    