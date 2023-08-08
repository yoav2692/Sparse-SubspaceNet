"""
    Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: Y. Amiel
    Created: 05/08/28
    Edited: 05/08/28

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for an experiment.
    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
# Imports
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.classes import *
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

class Commands():
    def __init__(self, save_experiment : bool = True, save_results : bool = True, create_data : bool = True, load_data : bool = True, load_model : bool = False, train_model : bool = True, save_model : bool = True, evaluate_mode : bool = True, plot_results: bool = False):
        self.save_experiment = save_experiment   # Saving experiment setup as python structure
        self.save_results  = save_results      # Saving results to file or present them over CMD
        self.create_data   = create_data       # Creating new dataset
        self.load_data     = load_data        # Loading data from exist dataset
        self.load_model    = load_model       # Load specific model for training
        self.train_model   = train_model       # Applying training operation
        self.save_model    = save_model        # Saving tuned model
        self.evaluate_mode = evaluate_mode     # Evaluating desired algorithms
        self.plot_results    = plot_results     # Evaluating desired algorithms

    def set_command(self,command,value):
        self.__setattr__(command,value)

class Opts(Enum):
    DEFAULT = "None"
    load = "load"
    save = "save"
    eval = "eval"
    plot = "plot"
    train = "train"
    create = "create"

class Framework():
    def __init__(self,name:str,commands:dict):
        self.name = name
        self.commands = Commands()

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
    def __init__(self, samples_size: int = 50000 , train_test_ratio: float = 0.05 , batch_size: int = 2048 , epochs: int = 40 , optimizer : str = "Adam" , learning_rate: float = 0.001 , weight_decay: float = 1e-9 , step_size: int = 80 , gamma: float = 0.2 , loss_method : Loss_method = Loss_method.DEFAULT.value , learning_curve_opt : bool = False):
        self.samples_size = samples_size  # Overall dateset size
        self.train_test_ratio = train_test_ratio  # training and testing datasets ratio
        self.batch_size= batch_size
        self.epochs= epochs
        self.optimizer= optimizer
        self.learning_rate= learning_rate
        self.weight_decay= weight_decay
        self.step_size= step_size
        self.gamma= gamma
        self.loss_method = loss_method
        self.learning_curve_opt = learning_curve_opt


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
    experiment_base = ExperimentSetup(
        framework= Framework(
            name= "Base",
            commands= Commands()
        ), 
        simulation_parameters=SimulationParams(
            sensors_array=SensorsArray("MRA-4"),#-virtualExtention-10
            signal_params= SignalParams(
                num_sources=2,
                num_observations=100,
                signal_type = Signal_type.narrowband.value , 
                signal_nature = Signal_nature.non_coherent.value
            ),
            noise_params= NoiseParams(
                snr = 10,
                sv_noise = 0,
                eta_sensors_dev = 0
            )
        ),
        algo_parameters= AlgoParams(
            training_params= TrainingParams(),
            preprocess_method = Model_type.SubspaceNet.value, # MatrixCompletion_spatialStationary",
            detection_method = detection_method.esprit.value,
            tau = 8
        )
    )

    '''
    TODO:
    Till ICASP:
        Imidate tasks:
            a. resources for faster computing: bottleneck - 
            b. "add phase" approach for missimg antennas (under far field assumption) - WIP
            c. limit source angle 
            c. Compare RMSPE with RMSE 
            d. Compare missing antennas fill options
        
        0. Plan:
            a. define experiments plan:
                MRA4:
                    1. sweep over: coherency , num sources
                    2. Add miscallibration
                    3. close targets
                    4. broad band
            b. Provide benchmarks from literature

        1. algo:
            a. Framework to compare 2 algo on the same data
            b. Improve spatial-statsionary: better than averaging
        
        2. Better DX ( low priority)
            a. exp documentable data set load
            b. complete ENUMing in Classes - add summerized defaults
            c. refactor to the same naming in all hierarchies
        
    September Onwards:
        0. MRA8 results

        1. Algo:
            a. Loss - define computationally cheap method that does not require permutation for sorted arrays 
                A. Stable match with multitask learning approach
            b. -- Utilize CS/low-Rank matrix complition
        
        3. missing antenna: 
            a. --fid sparseNet with partial data + array locations

        4. better DX:
            a. framework to find and analyze cases that caused high loss

        5. Further Research:
            a. High-Rank matrix complition w. Wasim
    '''

    experiment1 = experiment_base
    experiment1.framework.name = "ULA7_6sources_coherent"
    experiment1.simulation_parameters.sensors_array=SensorsArray("ULA-7")
    experiment1.simulation_parameters.signal_params.num_sources = 6
    experiment1.framework.commands.load_data = False
    experiment1.framework.commands.create_data = True
    experiment1.framework.commands.evaluate_mode = False
    experiment1.simulation_parameters.signal_params.signal_nature = Signal_nature.coherent.value
    experiment1.algo_parameters.training_params.samples_size = Dataset_size.pipe_cleaner.value
    experiment1.algo_parameters.training_params.epochs = Num_epochs.pipe_cleaner.value
    experiment1.algo_parameters.training_params.loss_method = Loss_method.no_permute.value
    main.run_experiment(experiment=experiment1)

    experiment_periodic = experiment1
    experiment_periodic.framework.name = "ULA7_6sources_coherent_check_periodic_loss"
    experiment_periodic.framework.commands.load_data = True
    experiment_periodic.framework.commands.create_data = False
    experiment_periodic.algo_parameters.training_params.loss_method = Loss_method.no_permute_periodic.value
    main.run_experiment(experiment=experiment_periodic)

    experiment_mra = experiment1
    experiment_mra.framework.name = "MRA5_permute_6sources_coherent"
    experiment_mra.framework.commands.load_data = False
    experiment_mra.framework.commands.create_data = True
    experiment_mra.simulation_parameters.sensors_array=SensorsArray("MRA-4",Missing_senors_handle_method.zeros.value)
    experiment_mra.simulation_parameters.signal_params.num_sources = 5
    experiment_mra.algo_parameters.training_params.loss_method = Loss_method.no_permute.value
    main.run_experiment(experiment=experiment_mra)

    experiment_mra_phase = experiment_mra
    experiment_mra.framework.name = "MRA5_permute_6sources_coherent_phase_continuetion"
    experiment_mra.framework.commands.load_data = True
    experiment_mra.framework.commands.create_data = False
    experiment_mra_phase.simulation_parameters.sensors_array=SensorsArray("MRA-4",Missing_senors_handle_method.phase_continuetion.value)

    experiment3 = experiment_mra
    experiment3.framework.name = "MRA5_permute_6sources_nonCoherent"
    experiment3.simulation_parameters.signal_params.signal_nature = Signal_nature.non_coherent.value
    main.run_experiment(experiment=experiment3)

    experiment_matrix_completion = experiment3
    experiment_matrix_completion.framework.name = "MRA5_matrixCompletion_6sources_nonCoherent"
    experiment_matrix_completion.framework.commands.load_data = True
    experiment_matrix_completion.framework.commands.create_data = False
    experiment_matrix_completion.framework.commands.train_model = False
    experiment_matrix_completion.algo_parameters.preprocess_method = Model_type.MatrixCompletion.value + "_" + matrix_completion_method.spatial_stationary.value
    experiment_matrix_completion.algo_parameters.training_params.loss_method = Loss_method.no_permute.value
    main.run_experiment(experiment=experiment_matrix_completion)
