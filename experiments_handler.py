"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 30/06/23

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for the simulation.
    This scripts calls the following functions:
        * create_dataset: For creating training and testing datasets 
        * training: For training DR-MUSIC model
        * evaluate_dnn_model: For evaluating subspace hybrid models

    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
# Imports
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.system_model import SystemModelParams
from src.signal_creation import *
from src.data_handler import *
from src.criterions import set_criterions
from src.training import *
from src.sensors_arrays import *
from src.evaluation import evaluate
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator
import main

# Initialization
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")
class Framework():
    def __init__(self,name:str,commands:dict):
        self.name = name
        self.commands = commands

class SignalParams():
    def __init__(self,num_sources : int, num_observations: int , signal_type: str, signal_nature: str):
        self.num_sources = num_sources
        self.num_observations = num_observations
        self.signal_type = signal_type
        self.signal_nature = signal_nature
                
class NoiseParams():
    def __init__(self, snr: float ,eta_sensors_dev:float = 0 , sv_noise: float = 0 ):
        self.snr = snr
        self.eta_sensors_dev = eta_sensors_dev
        self.sv_noise = sv_noise

class SimulationParams():
    def __init__(self, sensors_array: SensorsArray , signal_params:SignalParams , noise_params:NoiseParams):
        self.sensors_array = sensors_array
        self.signal_params = signal_params
        self.noise_params = noise_params

class TrainingParams():
    def __init__(self, samples_size: int = 50000 , train_test_ratio: float = 0.05 , batch_size: int = 2048 , epochs: int = 40 , optimizer : str = "Adam" , learning_rate: float = 0.00001 , weight_decay: float = 1e-9 , step_size: int = 80 , gamma: float = 0.2):
        self.samples_size = samples_size  # Overall dateset size
        self.train_test_ratio = train_test_ratio  # training and testing datasets ratio
        self.batch_size= batch_size
        self.epochs= epochs
        self.optimizer= optimizer
        self.learning_rate= learning_rate
        self.weight_decay= weight_decay
        self.step_size= step_size
        self.gamma= gamma

class AlgoParams():
    def __init__(self , training_params:TrainingParams , preprocess_method : str, detection_method: str , tau: str):
        self.training_params = training_params
        self.preprocess_method = preprocess_method
        self.detection_method = detection_method
        self.tau = tau

class ExperimentSetup():
    def __init__(self , framework: Framework , simulation_parameters:SimulationParams ,algo_parameters: AlgoParams):
        self.framework = framework
        self.simulation_parameters = simulation_parameters
        self.algo_parameters = algo_parameters

if __name__ == "__main__":
    experiment1 = ExperimentSetup(
        framework= Framework(
            name= "T1",
            commands= {
                "SAVE_EXPERIMENT": True,  # Saving experiment setup as python structure
                "SAVE_RESULTS": True,  # Saving results to file or present them over CMD
                "CREATE_DATA": True,  # Creating new dataset
                "LOAD_DATA": True,  # Loading data from exist dataset
                "LOAD_MODEL": False,  # Load specific model for training
                "TRAIN_MODEL": True,  # Applying training operation
                "SAVE_MODEL": True,  # Saving tuned model
                "EVALUATE_MODE": True,  # Evaluating desired algorithms
            }
        ), 
        simulation_parameters=SimulationParams(
            sensors_array=SensorsArray("MRA-4"),#-virtualExtention-10
            signal_params= SignalParams(
                num_sources=2,
                num_observations=200,
                signal_type = "NarrowBand",
                signal_nature = "non-coherent"
            ),
            noise_params= NoiseParams(
                snr = 10,
                sv_noise = 0,
                eta_sensors_dev = 0
            )
        ),
        algo_parameters= AlgoParams(
            training_params= TrainingParams(),
            preprocess_method = "MatrixCompletion_spatialStationary",
            detection_method = "esprit",
            tau = 8
        )
    )
    experiment2 = experiment1
    experiment2.framework.name = "POC"
    experiment2.simulation_parameters.sensors_array=SensorsArray("ULA-8")
    experiment2.algo_parameters.preprocess_method = "SubspaceNet"
    experiment2.algo_parameters.training_params.samples_size = 10000
    main.run_experiment(experiment=experiment2)
    experiment3 = experiment2
    experiment3.framework.commands["LOAD_DATA"] = True