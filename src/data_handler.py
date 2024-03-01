"""Subspace-Net 
Details
----------
    Name: data_handler.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 03/06/23

Purpose:
--------
    This scripts handle the creation and processing of synthetic datasets
    based on specified parameters and model types.
    It includes functions for generating datasets, reading data from files,
    computing autocorrelation matrices, and creating covariance tensors.

Attributes:
-----------
    Samples (from src.signal_creation): A class for creating samples used in dataset generation.

    The script defines the following functions:
    * create_dataset: Generates a synthetic dataset based on the specified parameters and model type.
    * read_data: Reads data from a file specified by the given path.
    * autocorrelation_matrix: Computes the autocorrelation matrix for a given lag of the input samples.
    * create_autocorrelation_tensor: Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.
    * create_cov_tensor: Creates a 3D tensor containing the real part,
        imaginary part, and phase component of the covariance matrix.
    * set_dataset_filename: Returns the generic suffix of the datasets filename.

"""

# Imports
import torch
# import cvxpy as cp
import numpy as np
import itertools
from tqdm import tqdm
from src.sensors_arrays import SensorsArray
from src.signal_creation import Samples
from src.classes import *
from pathlib import Path
from src.system_model import SystemModelParams
from src.correlation import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ANGLE_LIM = 70

def create_dataset(
    system_model_params: SystemModelParams,
    samples_size: int,
    model_type: str,
    tau: int = None,
    save_datasets: bool = False,
    datasets_path: Path = None,
    true_doa: list = None,
    phase: str = None,
):
    """
    Generates a synthetic dataset based on the specified parameters and model type.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams
        samples_size (float): The size of the dataset.
        tau (int): The number of lags for auto-correlation (relevant only for SubspaceNet model).
        model_type (str): The type of the model.
        save_datasets (bool, optional): Specifies whether to save the dataset. Defaults to False.
        datasets_path (Path, optional): The path for saving the dataset. Defaults to None.
        true_doa (list, optional): Predefined angles. Defaults to None.
        phase (str, optional): The phase of the dataset (test or training phase for CNN model). Defaults to None.

    Returns:
    --------
        tuple: A tuple containing the desired dataset comprised of (X-samples, Y-labels).

    """
    generic_dataset = []
    samples_model = Samples(system_model_params)
    # Generate permutations for CNN model training dataset
    if model_type.startswith(Model_type.DeepCNN.value) and phase.startswith("train"):
        doa_permutations = []
        angles_grid = np.linspace(start=-ANGLE_LIM, stop=ANGLE_LIM, num=361)
        for comb in itertools.combinations(angles_grid, system_model_params.M):
            doa_permutations.append(list(comb))

    if model_type.startswith(Model_type.DeepCNN.value) and phase.startswith("train"):
        for i, doa in tqdm(enumerate(doa_permutations)):
            # Samples model creation
            samples_model.set_doa(doa)
            # Observations matrix creation
            X = torch.tensor(
                samples_model.samples_creation(
                    noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
                )[0],
                dtype=torch.complex64,
            )
            # Ground-truth creation
            Y = torch.zeros_like(torch.tensor(angles_grid))
            for angle in doa:
                Y[list(angles_grid).index(angle)] = 1
            generic_dataset.append((X, Y))
    else:
        for i in tqdm(range(samples_size)):
            # Samples model creation
            samples_model.set_doa(true_doa)
            # Observations matrix creation
            X = torch.tensor(
                samples_model.samples_creation(
                    noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
                )[0],
                dtype=torch.complex64,
            )
            # Ground-truth creation
            Y = torch.tensor(samples_model.doa, dtype=torch.float64)
            generic_dataset.append((X, Y))

    if save_datasets:
        generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(
            system_model_params, samples_size
        )
        samples_model_filename = f"samples_model" + set_dataset_filename(
            system_model_params, samples_size
        )

        torch.save(
            obj=generic_dataset, f=datasets_path / phase / generic_dataset_filename
        )
        if phase.startswith("test"):
            torch.save(
                obj=samples_model, f=datasets_path / phase / samples_model_filename
            )

    return generic_dataset, samples_model

def covariance_handler(X, tau):
    if model_type.startswith(Model_type.SubspaceNet.value):
        # Generate auto-correlation tensor
        Rx = create_autocorrelation_tensor(X, tau).to(torch.float)
    elif model_type.startswith(Model_type.DeepCNN.value) and phase.startswith("test"):
        # Generate 3d covariance parameters tensor
        Rx = create_cov_tensor(X)
    else:
        Rx = X
    return Rx

def feature_extraction(X,sensors_array : SensorsArray, tau:int):
    features = []
    for x in X:
        x = sample_missing_sensors_handle(samples = x, sensors_array = sensors_array)
        features.append(create_autocorrelation_tensor(x,tau).to(torch.float))
    return torch.stack(features, dim=0)

# def read_data(Data_path: str) -> torch.Tensor:
def read_data(path: str):
    """
    Reads data from a file specified by the given path.

    Args:
    -----
        path (str): The path to the data file.

    Returns:
    --------
        torch.Tensor: The loaded data.

    Raises:
    -------
        None

    Examples:
    ---------
        >>> path = "data.pt"
        >>> read_data(path)

    """
    assert isinstance(path, (str, Path))
    data = torch.load(path)
    return data

def sample_missing_sensors_handle(samples , sensors_array : SensorsArray):
    expansion_matrix = init_expansion_matrix(sensors_array , samples.type())
    samples = torch.matmul(expansion_matrix , samples[sensors_array.locs])
    return samples

def init_expansion_matrix(sensors_array : SensorsArray , type):
    phase_continuation_expansion_matrix = np.zeros((sensors_array.last_sensor_loc,len(sensors_array.locs)))
    list_sensors_array_locs = list(sensors_array.locs)
    for sensor_loc in range(sensors_array.last_sensor_loc):
        if sensor_loc in sensors_array.locs:
            sensor_loc_ind = list_sensors_array_locs.index(sensor_loc)
            phase_continuation_expansion_matrix[sensor_loc][sensor_loc_ind] = 1
        else:
            if sensors_array.missing_sensors_handle_method == Missing_senors_handle_method.phase_continuation.value:
                diffs = sensors_array.locs - sensor_loc
                phase_diff = diffs[np.argmin(abs(diffs))]
                closest_sensor = sensors_array.locs[np.argmin(abs(diffs))]
                closest_sensor_loc_ind = list_sensors_array_locs.index(closest_sensor)
                phase_continuation_expansion_matrix[sensor_loc][closest_sensor_loc_ind] = np.exp(
                    -1j * np.pi * phase_diff
                )
            else: # Missing_senors_handle_method.zeros.value
                pass
    return torch.tensor(phase_continuation_expansion_matrix , dtype=torch.complex64 , requires_grad=True)

def load_datasets(
    system_model_params: SystemModelParams,
    model_type: str,
    samples_size: float,
    datasets_path: Path,
    train_test_ratio: float,
    is_training: bool = False,
):
    """
    Load different datasets based on the specified parameters and phase.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        model_type (str): The type of the model.
        samples_size (float): The size of the overall dataset.
        datasets_path (Path): The path to the datasets.
        train_test_ratio (float): The ration between train and test datasets.
        is_training (bool): Specifies whether to load the training dataset.

    Returns:
    --------
        List: A list containing the loaded datasets.

    """
    datasets = []
    # Define test set size
    test_samples_size = int(train_test_ratio * samples_size)
    # Generate datasets filenames
    generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(
        system_model_params, test_samples_size
    )
    samples_model_filename = f"samples_model" + set_dataset_filename(
        system_model_params, test_samples_size
    )

    # Whether to load the training dataset
    if is_training:
        # Load training dataset
        try:
            model_trainingset_filename = f"Generic_DataSet" + set_dataset_filename(
                system_model_params, samples_size
            )
            generic_train_dataset = read_data(
                datasets_path / "train" / model_trainingset_filename
            )
            datasets.append(generic_train_dataset)
        except:
            raise Exception("load_datasets: Training dataset doesn't exist")
    # Load generic test dataset
    try:
        generic_test_dataset = read_data(
            datasets_path / "test" / generic_dataset_filename
        )
        datasets.append(generic_test_dataset)
    except:
        raise Exception("load_datasets: Generic test dataset doesn't exist")
    # Load samples models
    try:
        samples_model = read_data(datasets_path / "test" / samples_model_filename)
        datasets.append(samples_model)
    except:
        raise Exception("load_datasets: Samples model dataset doesn't exist")
    return datasets


def set_dataset_filename(system_model_params: SystemModelParams, samples_size: float):
    """Returns the generic suffix of the datasets filename.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        samples_size (float): The size of the overall dataset.

    Returns:
    --------
        str: Suffix dataset filename
    """
    suffix_filename = (
        f"_{system_model_params.signal_type}_"
        + f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_"
        + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
        + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}_"
        + f"sensors_array_form={system_model_params.sensors_array_form}"
        + ".h5"
    )
    return suffix_filename
