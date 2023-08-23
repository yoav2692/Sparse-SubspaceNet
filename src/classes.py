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
from dataclasses import dataclass


class Opts(Enum):
    DEFAULT = "None"
    load = "Load"
    save = "Save"
    eval = "Eval"
    plot = "Plot"
    train = "Train"
    create = "Create"

class Commands:
    def __init__(
        self,
        save_experiment: bool = True,
        save_results: bool = True,
        create_data: bool = True,
        load_data: bool = True,
        load_model: bool = False,
        train_model: bool = True,
        save_model: bool = True,
        evaluate_mode: bool = True,
        plot_results: bool = False,
    ):
        self.save_experiment = (
            save_experiment  # Saving experiment setup as python structure
        )
        self.save_results = (
            save_results  # Saving results to file or present them over CMD
        )
        self.create_data = create_data  # Creating new dataset
        self.load_data = load_data  # Loading data from exist dataset
        self.load_model = load_model  # Load specific model for training
        self.train_model = train_model  # Applying training operation
        self.save_model = save_model  # Saving tuned model
        self.evaluate_mode = evaluate_mode  # Evaluating desired algorithms
        self.plot_results = plot_results  # Evaluating desired algorithms

    def set_data_opts(self, opt: str):
        self.create_data = Opts.create.value in opt
        self.load_data = Opts.load.value in opt

    def set_model_opts(self, opt: str):
        self.load_model = Opts.load.value in opt
        self.train_model = Opts.train.value in opt
        self.save_model = Opts.save.value in opt
        self.evaluate_mode = Opts.eval.value in opt

    def set_results_opts(self, opt: str):
        self.save_results = Opts.save.value in opt
        self.plot_results = Opts.plot.value in opt


class Missing_senors_handle_method(Enum):
    DEFAULT = "zeros"
    zeros = "zeros"
    phase_continuation = "phase_continuation"
    remove = "remove"
    noise = "noise"


class Signal_nature(Enum):
    DEFAULT = "non-coherent"
    coherent = "coherent"
    non_coherent = "non-coherent"


class Signal_type(Enum):
    DEFAULT = "NarrowBand"
    narrowband = "NarrowBand"
    broadband = "BroadBand"


class Num_observations(Enum):
    DEFAULT = int(100)
    huge = int(1000)
    big = int(200)
    normal = int(100)
    small = int(30)


class Model_type(Enum):
    DEFAULT = "SubspaceNet"
    SubspaceNet = "SubspaceNet"
    MatrixCompletion = "MatrixCompletion"
    DeepCNN = "DeepCNN"


class matrix_completion_method(Enum):
    spatial_stationary = "spatial_stationary"
    low_rank = "low_rank"


class detection_method(Enum):
    DEFAULT = "esprit"
    esprit = "esprit"
    root_music = "root_music"


class Criterion(Enum):
    DEFAULT = "From_Model_name"
    RMSE = "RMSE"
    RMSPE = "RMSPE"
    BCE = "BCE"


class Loss_method(Enum):
    DEFAULT = "no_permute"
    sort = "sort"
    no_permute = "no_permute"
    full_permute = "full_permute"
    no_permute_periodic = "no_permute_periodic"  # temp for POC
    full_permute_periodic = "full_permute_periodic"
    # closest_with_repetiotions = "closest_with_repetiotions"
    # stable_match = "stable_match"


class Optimizer(Enum):
    DEFAULT = "Adam"
    Adam = "Adam"
    SGD = "SGD"
    SGD_Momentum = "SGD Momentum"


class Dataset_size(Enum):
    DEFAULT = int(50e3)
    test = int(100e3)
    normal = int(50e3)
    small = int(10e3)
    pipe_cleaner = int(100)


class Num_epochs(Enum):
    DEFAULT = int(40)
    test = int(80)
    normal = int(40)
    small = int(20)
    pipe_cleaner = int(2)
