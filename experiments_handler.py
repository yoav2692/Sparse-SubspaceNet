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
from lib2to3.pgen2.token import OP
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.classes import *
from src.system_model import SystemModelParams
from src.signal_creation import *
from src.data_handler import *
from src.training import *
from src.sensors_arrays import *
from src.evaluation import evaluate
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator
import main


class Framework:
    def __init__(self, name: str):
        self.name = name
        self.commands = Commands()


class SignalParams:
    def __init__(
        self,
        num_sources: int,
        num_observations: int,
        signal_type: str,
        signal_nature: str,
        doa_range: float,
        doa_gap: float,
        multi_num_sources_flag: bool  = False,
        is_known_num_sources: bool  = True
    ):
        self.num_sources = num_sources
        self.num_observations = num_observations
        self.signal_type = signal_type
        self.signal_nature = signal_nature
        self.doa_range = doa_range
        self.doa_gap = doa_gap
        self.multi_num_sources_flag = multi_num_sources_flag
        self.is_known_num_sources = is_known_num_sources


class NoiseParams:
    def __init__(self, snr: float, eta_sensors_dev: float = 0, sv_noise: float = 0):
        self.snr = snr
        self.eta_sensors_dev = eta_sensors_dev
        self.sv_noise = sv_noise


class SimulationParams:
    def __init__(
        self,
        sensors_array: SensorsArray,
        signal_params: SignalParams,
        noise_params: NoiseParams,
    ):
        self.sensors_array = sensors_array
        self.signal_params = signal_params
        self.noise_params = noise_params


class TrainingParams:
    def __init__(
        self,
        samples_size: int = Dataset_size.DEFAULT.value,
        train_test_ratio: float = 0.05,
        batch_size: int = 2048,
        epochs: int = Num_epochs.DEFAULT.value,
        optimizer: str = Optimizer.DEFAULT.value,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-9,
        step_size: int = 80,
        gamma: float = 0.2,
        loss_method: Loss_method = Loss_method.DEFAULT.value,
        criterion_name: str = Criterion.DEFAULT.value,
        learning_curve_opt: bool = False,
    ):
        self.samples_size = samples_size  # Overall dateset size
        self.train_test_ratio = train_test_ratio  # training and testing datasets ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.loss_method = loss_method
        self.criterion_name = criterion_name
        self.learning_curve_opt = learning_curve_opt

    def set_train_time(self, opt: str = "", samples_size: int = 0, epochs: int = 0):
        if opt:
            self.samples_size = Dataset_size[opt].value
            self.epochs = Num_epochs[opt].value
        if samples_size:
            self.samples_size = samples_size
        if epochs:
            self.epochs = epochs


class EvaluationParams:
    def __init__(
        self,
        samples_size: int = Dataset_size.DEFAULT.value,
        train_test_ratio: float = 0.05,
        batch_size: int = 2048,
        epochs: int = Num_epochs.DEFAULT.value,
        optimizer: str = Optimizer.DEFAULT.value,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-9,
        step_size: int = 80,
        gamma: float = 0.2,
        loss_method: Loss_method = Loss_method.DEFAULT.value,
        criterion_name: str = Criterion.DEFAULT.value,
        learning_curve_opt: bool = False
    ):
        self.samples_size = samples_size  # Overall dateset size
        self.loss_method = loss_method
        self.criterion_name = criterion_name


class AlgoParams:
    def __init__(
        self,
        training_params: TrainingParams,
        evaluation_params: EvaluationParams,
        preprocess_method: str,
        detection_method: str,
        tau: str,
    ):
        self.training_params = training_params
        self.evaluation_params = evaluation_params
        self.preprocess_method = preprocess_method
        self.detection_method = detection_method
        self.tau = tau


class ExperimentSetup:
    def __init__(
        self,
        framework: Framework,
        simulation_parameters: SimulationParams,
        algo_parameters: AlgoParams,
    ):
        self.framework = framework
        self.simulation_parameters = simulation_parameters
        self.algo_parameters = algo_parameters

    """
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
    """

    def experiment_sweep(num_sources):
        experiment_base = ExperimentSetup(
            framework=Framework(name="Base"),
            simulation_parameters=SimulationParams(
                sensors_array=SensorsArray("MRA-4"),  # -virtualExtention-10
                signal_params=SignalParams(
                    num_sources=num_sources,
                    num_observations=Num_observations.big.value,
                    signal_type=Signal_type.narrowband.value,
                    signal_nature=Signal_nature.non_coherent.value,
                    doa_gap=15,
                    doa_range=180,
                    
                ),
                noise_params=NoiseParams(snr=10, sv_noise=0, eta_sensors_dev=0),
            ),
            algo_parameters=AlgoParams(
                training_params=TrainingParams(),
                evaluation_params=EvaluationParams(),
                preprocess_method=Model_type.SubspaceNet.value,  # MatrixCompletion_spatialStationary",
                detection_method=detection_method.esprit.value,
                tau=8,
            ),
        )
        # Dor Editing Section
        for num_sources in [6]:
            experiment_ula = experiment_base
            experiment_ula.framework.name = (
                f"MRA-4_{num_sources}_spacing_mis-calibration"
            )
            # experiment_ula.simulation_parameters.sensors_array = SensorsArray("ULA-7")
            experiment_ula.simulation_parameters.sensors_array=SensorsArray("MRA-4")
            experiment_ula.simulation_parameters.signal_params.num_sources = num_sources
            experiment_ula.framework.commands.set_data_opts(Opts.create.value)
            experiment_ula.framework.commands.set_model_opts(
                Opts.train.value + Opts.eval.value + Opts.save.value
            )
            experiment_ula.framework.commands.set_results_opts(Opts.save.value)
            experiment_ula.simulation_parameters.signal_params.signal_nature = (
                Signal_nature.non_coherent.value
            )
            experiment_ula.simulation_parameters.signal_params.doa_range = 140
            experiment_ula.simulation_parameters.signal_params.doa_gap = 10
            experiment_ula.algo_parameters.training_params.learning_rate = 0.00001
            experiment_ula.algo_parameters.training_params.set_train_time(
                samples_size=70000, epochs=40
            )
            experiment_ula.algo_parameters.training_params.step_size = 80
            experiment_ula.algo_parameters.training_params.gamma = 0.2
            experiment_ula.algo_parameters.training_params.loss_method = (
                Loss_method.no_permute.value
            )
            experiment_ula.algo_parameters.training_params.criterion_name = (
                Criterion.RMSE.value
            )
            experiment_ula.algo_parameters.evaluation_params.criterion_name = (
                Criterion.RMSE.value
            )
            main.run_experiment(experiment=experiment_ula)


if __name__ == "__main__":
    debug = 1
    if debug:
        ExperimentSetup.experiment_sweep(6)
    else:
        user_input = " ".join(sys.argv[1:])
        num_sources = int(user_input[0])
        ExperimentSetup.experiment_sweep(num_sources)
