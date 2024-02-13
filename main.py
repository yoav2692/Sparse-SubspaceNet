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
import pickle
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

# Initialization
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")


def run_experiment(experiment):
    # Initialize paths
    os.system(f"echo start {experiment.framework.name}")
    external_data_path = Path.cwd() / "data"
    datasets_path = external_data_path / "datasets"
    simulations_path = external_data_path / "simulations"
    saving_path = external_data_path / "weights"
    # Initialize time and date
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = experiment.framework.name + "_" + now.strftime("%d_%m_%Y_%H_%M")
    # Operations commands
    commands = experiment.framework.commands
    # Saving simulation scores to external file
    if commands.save_experiment:
        file_path = (
            simulations_path / "experiments" / Path(dt_string_for_save + ".pkl")
        )
        afile = open(file_path, "wb")
        pickle.dump(experiment, afile)
        afile.close()
    if commands.save_results:
        file_path = (
            simulations_path / "results" / "scores" / Path(dt_string_for_save + ".txt")
        )
        sys.stdout = open(file_path, "w")
    
    # Define system model parameters
    system_model_params = (
        SystemModelParams()
        .set_sensors_array(experiment.simulation_parameters.sensors_array)
        .set_num_sources(experiment.simulation_parameters.signal_params.num_sources)
        .set_num_observations(experiment.simulation_parameters.signal_params.num_observations)
        .set_signal_type(experiment.simulation_parameters.signal_params.signal_type)
        .set_signal_nature(experiment.simulation_parameters.signal_params.signal_nature)
        .set_doa_range(experiment.simulation_parameters.signal_params.doa_range)
        .set_doa_gap(experiment.simulation_parameters.signal_params.doa_gap)
        .set_snr(experiment.simulation_parameters.noise_params.snr)
        .set_sensors_dev(experiment.simulation_parameters.noise_params.eta_sensors_dev)
        .set_sv_noise(experiment.simulation_parameters.noise_params.sv_noise)
        .set_multi_num_sources_flag(experiment.simulation_parameters.signal_params.multi_num_sources_flag)
        .set_is_known_num_sources(experiment.simulation_parameters.signal_params.is_known_num_sources)
    )
    # Generate model configuration
    model_config = (
        ModelGenerator()
        .set_model_type(experiment.algo_parameters.preprocess_method)
        .set_diff_method(experiment.algo_parameters.detection_method)
        .set_tau(experiment.algo_parameters.tau)
        .set_model(system_model_params)
    )
    # Define samples size
    samples_size = experiment.algo_parameters.training_params.samples_size  # Overall dateset size
    train_test_ratio = experiment.algo_parameters.training_params.train_test_ratio  # training and testing datasets ratio
    # Sets simulation filename
    simulation_filename = get_simulation_filename(
        system_model_params=system_model_params, model_config=model_config
    )
    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    # Initialize seed
    set_unified_seed()
    # Datasets creation
    if commands.create_data:
        # Define which datasets to generate
        create_training_data = True  # Flag for creating training data
        create_testing_data = True  # Flag for creating test data
        print("Creating Data...")
        if create_training_data:
            # Generate training dataset
            generic_train_dataset, _ = create_dataset(
                system_model_params=system_model_params,
                samples_size=samples_size,
                model_type=model_config.model_type,
                tau=model_config.tau,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=None,
                phase="train",
            )
            # Generate test dataset
            generic_test_dataset, test_samples_model = create_dataset(
                system_model_params=system_model_params,
                samples_size=int(train_test_ratio * samples_size),
                model_type=model_config.model_type,
                tau=model_config.tau,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=None,
                phase="test",
            )
    # Datasets loading
    elif commands.load_data:
        (
            generic_train_dataset,
            generic_test_dataset,
            test_samples_model,
        ) = load_datasets(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            samples_size=samples_size,
            datasets_path=datasets_path,
            train_test_ratio=train_test_ratio,
            is_training=True,
        )

    # Training stage
    if commands.train_model:
        # Assign the training parameters object
        simulation_parameters = (
            TrainingParams()
            .set_batch_size(experiment.algo_parameters.training_params.batch_size)
            .set_epochs(experiment.algo_parameters.training_params.epochs)
            # .set_model(system_model=test_samples_model, tau=model.tau, diff_method=model.diff_method)
            .set_model(model=model_config)
            .set_optimizer(optimizer=experiment.algo_parameters.training_params.optimizer, learning_rate=experiment.algo_parameters.training_params.learning_rate, weight_decay=experiment.algo_parameters.training_params.weight_decay)
            .set_training_dataset(generic_train_dataset)
            .set_schedular(step_size=experiment.algo_parameters.training_params.step_size, gamma=experiment.algo_parameters.training_params.gamma)
            .set_criterion(criterion_name= experiment.algo_parameters.training_params.criterion_name , loss_method=experiment.algo_parameters.training_params.loss_method)
        )
        if commands.load_model:
            simulation_parameters.load_model(
                loading_path=saving_path / "final_models" / simulation_filename
            )
        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            sensors_array = system_model_params.sensors_array,
            model_name=simulation_filename,
            saving_path=saving_path,
            plot_curves = commands.plot_results
        )
        # Save model weights
        if commands.save_model:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands.save_results and commands.plot_results:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        elif commands.plot_results:
            plt.show()

    # Evaluation stage
    if commands.evaluate_mode:
        # Initialize figures dict for plotting
        figures = initialize_figures()
        # Define loss measure for evaluation
        criterion, subspace_criterion = set_criterions(str(experiment.algo_parameters.evaluation_params.criterion_name))
        # Load datasets for evaluation
        if not (commands.create_data or commands.load_data):
            generic_test_dataset, test_samples_model = load_datasets(
                system_model_params=system_model_params,
                model_type=model_config.model_type,
                samples_size=samples_size,
                datasets_path=datasets_path,
                train_test_ratio=train_test_ratio,
            )

        # Generate DataLoader objects
        generic_test_dataset = torch.utils.data.DataLoader(
            generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        # Load pre-trained model
        if not commands.train_model:
            # Define an evaluation parameters instance
            simulation_parameters = (
                TrainingParams()
                .set_model(model=model_config)
                .load_model(
                    loading_path=saving_path / "final_models" / simulation_filename
                )
            )
            model = simulation_parameters.model
        # print simulation summary details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            phase="evaluation",
            parameters=simulation_parameters,
        )
        # Evaluate DNN models, augmented and subspace methods
        evaluate(
            model=model,
            model_type=model_config.model_type,
            sensors_array = system_model_params.sensors_array,
            tau=model_config.tau,
            generic_test_dataset=generic_test_dataset,
            criterion=criterion,
            subspace_criterion=subspace_criterion,
            system_model=test_samples_model,
            figures=figures,
            plot_spec=False,
        )
    if commands.plot_results:
        plt.show()
    print("end")
