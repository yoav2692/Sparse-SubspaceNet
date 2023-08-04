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
from src.signal_creation import Samples
from src.sensors_arrays import MRA_LOCS , MRA_VIRTUAL_ANTS
from pathlib import Path
from src.system_model import SystemModelParams

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataset(
    system_model_params: SystemModelParams,
    samples_size: float,
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
    model_dataset = []
    samples_model = Samples(system_model_params)
    # Generate permutations for CNN model training dataset
    if model_type.startswith("DeepCNN") and phase.startswith("train"):
        doa_permutations = []
        angles_grid = np.linspace(start=-90, stop=90, num=361)
        for comb in itertools.combinations(angles_grid, system_model_params.M):
            doa_permutations.append(list(comb))

    if model_type.startswith("DeepCNN") and phase.startswith("train"):
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
            X_model = create_cov_tensor(X)
            # Ground-truth creation
            Y = torch.zeros_like(torch.tensor(angles_grid))
            for angle in doa:
                Y[list(angles_grid).index(angle)] = 1
            model_dataset.append((X_model, Y))
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
            if model_type.startswith("SubspaceNet"):
                # Generate auto-correlation tensor
                X_model = create_autocorrelation_tensor(X, tau).to(torch.float)
            elif model_type.startswith("DeepCNN") and phase.startswith("test"):
                # Generate 3d covariance parameters tensor
                X_model = create_cov_tensor(X)
            elif model_type.startswith("MatrixCompletion"):
                # Generate auto-correlation tensor
                tau = 6 # TODO fix
                matrix_completion = {"method": "_".join(model_type.rsplit('_')[1:]) , "calc_cov_ants":MRA_LOCS[system_model_params.sparse_form]}
                X_model = create_autocorrelation_tensor(X[MRA_LOCS[system_model_params.sparse_form]], tau  ,matrix_completion ).to(torch.float)
            else:
                X_model = X
            # Ground-truth creation
            Y = torch.tensor(samples_model.doa, dtype=torch.float64)
            generic_dataset.append((X, Y))
            model_dataset.append((X_model, Y))

    if save_datasets:
        model_dataset_filename = f"{model_type}_DataSet" + set_dataset_filename(
            system_model_params, samples_size
        )
        generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(
            system_model_params, samples_size
        )
        samples_model_filename = f"samples_model" + set_dataset_filename(
            system_model_params, samples_size
        )

        torch.save(obj=model_dataset, f=datasets_path / phase / model_dataset_filename)
        torch.save(
            obj=generic_dataset, f=datasets_path / phase / generic_dataset_filename
        )
        if phase.startswith("test"):
            torch.save(
                obj=samples_model, f=datasets_path / phase / samples_model_filename
            )

    return model_dataset, generic_dataset, samples_model


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


# def autocorrelation_matrix(X: torch.Tensor, lag: int) -> torch.Tensor:
def autocorrelation_matrix(X: torch.Tensor, lag: int , matrix_completion: dict ):
    """
    Computes the autocorrelation matrix for a given lag of the input samples.

    Args:
    -----
        X (torch.Tensor): Samples matrix input with shape [N, T].
        lag (int): The requested delay of the autocorrelation calculation.

    Returns:
    --------
        torch.Tensor: The autocorrelation matrix for the given lag.

    """
    Rx_lag = torch.zeros(X.shape[0], X.shape[0], dtype=torch.complex128).to(device)
    for t in range(X.shape[1] - lag):
        # meu = torch.mean(X,1)
        x1 = torch.unsqueeze(X[:, t], 1).to(device)
        x2 = torch.t(torch.unsqueeze(torch.conj(X[:, t + lag]), 1)).to(device)
        Rx_lag += torch.matmul(x1 - torch.mean(X), x2 - torch.mean(X)).to(device)
    if matrix_completion:
        if matrix_completion["method"] == "spatialStationary":
            Rx_lag = torch.from_numpy(spatial_stationary_matrix_complition(matrix_completion["calc_cov_ants"],Rx_lag))
        elif matrix_completion["method"] == "lowRank" :
            Rx_lag = torch.from_numpy(low_rank_matrix_complition(matrix_completion["calc_cov_ants"],Rx_lag))
        else:
            raise Exception(f"Unknown matrix completion method: {matrix_completion['method']}")
            
    Rx_lag = Rx_lag / (X.shape[-1] - lag)
    Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), 0)
    return Rx_lag


# def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
def create_autocorrelation_tensor(X: torch.Tensor, tau: int , matrix_completion: dict = {}):
    """
    Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (BS, N, T).
        tau (int): Maximal time difference for the autocorrelation tensor.

    Returns:
    --------
        torch.Tensor: Tensor containing all the autocorrelation matrices,
                    with size (Batch size, tau, 2N, N).

    Raises:
    -------
        None

    """
    Rx_tau = []
    for i in range(tau):
        Rx_tau.append(autocorrelation_matrix(X, lag=i , matrix_completion = matrix_completion ))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr


def spatial_stationary_matrix_complition(array_locations , cov_matrix):
    """
    Complete the covariance matrix assuming spatial stationary.
    Diagonals of the complete matrix = average over same difference in the sparse matrix

    Args:
    -----
        X (np.ndarray): Input samples matrix.

    Returns:
    --------
        covariance_mat (np.ndarray): Covariance matrix.
    """
    virtual_size = array_locations[-1] + 1
    virtual_cov_matrix = np.zeros((virtual_size,virtual_size), dtype=complex)
    naive_cov_matrix_val = np.zeros(virtual_size, dtype=complex)
    naive_cov_matrix_elements = np.zeros(virtual_size)
    # other diag
    for array_loc_low_ind , array_loc_low in enumerate(array_locations):
        for array_loc_high_ind , array_loc_high in enumerate(array_locations[array_loc_low_ind+1:]):
            diff = array_loc_high - array_loc_low
            virtual_cov_matrix[array_loc_low,array_loc_high] = cov_matrix[array_loc_low_ind,array_loc_high_ind]
            naive_cov_matrix_val[diff] += cov_matrix[array_loc_low_ind,array_loc_high_ind]
            naive_cov_matrix_elements[diff] += 1
    naive_cov_matrix_val = [naive_cov_matrix_val[i]/naive_cov_matrix_elements[i] if naive_cov_matrix_elements[i] else 0 for i in range(virtual_size)]
    for diff in range(1,virtual_size):
        for m in range(virtual_size):
            for m_tag in range(m+1,virtual_size):
                if m in array_locations and m_tag in array_locations:
                    continue
                elif m_tag - m == diff :
                    virtual_cov_matrix[m,m_tag] = naive_cov_matrix_val[diff]

    # hermitian assumption
    virtual_cov_matrix += virtual_cov_matrix.T.conj()

    # main diag
    for m in range(virtual_size):
        if m in array_locations:
            loc = array_locations.index(m)
            virtual_cov_matrix[m,m] = cov_matrix[loc,loc]
        else:
            virtual_cov_matrix[m,m] =  naive_cov_matrix_val[0]
    return virtual_cov_matrix

def low_rank_matrix_complition(array_locations , cov_matrix):
    virtual_size = array_locations[-1] + 1
    virtual_cov_matrix = np.zeros((virtual_size,virtual_size), dtype=complex)
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((virtual_size,virtual_size), hermitian=True)
    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    # Constrain per pair of indices in the original array
    for array_loc_1_ind , array_loc_1 in enumerate(array_locations):
        for array_loc_2_ind , array_loc_2 in enumerate(array_locations):
            constraints.append(X[array_loc_1][array_loc_2] == cov_matrix[array_loc_1_ind][array_loc_2_ind])
    # Nuclear norm minimization
    prob = cp.Problem(cp.Minimize(cp.trace(X)),constraints)
    prob.solve()
    virtual_cov_matrix = X.value
    return virtual_cov_matrix

# def create_cov_tensor(X: torch.Tensor) -> torch.Tensor:
def create_cov_tensor(X: torch.Tensor):
    """
    Creates a 3D tensor of size (NxNx3) containing the real part, imaginary part, and phase component of the covariance matrix.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (N, T).

    Returns:
    --------
        Rx_tensor (torch.Tensor): Tensor containing the auto-correlation matrices, with size (Batch size, N, N, 3).

    Raises:
    -------
        None

    """
    Rx = torch.cov(X)
    Rx_tensor = torch.stack((torch.real(Rx), torch.imag(Rx), torch.angle(Rx)), 2)
    return Rx_tensor


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
    model_dataset_filename = f"{model_type}_DataSet" + set_dataset_filename(
        system_model_params, test_samples_size
    )
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
            model_trainingset_filename = f"{model_type}_DataSet" + set_dataset_filename(
                system_model_params, samples_size
            )
            train_dataset = read_data(
                datasets_path / "train" / model_trainingset_filename
            )
            datasets.append(train_dataset)
        except:
            raise Exception("load_datasets: Training dataset doesn't exist")
    # Load test dataset
    try:
        test_dataset = read_data(datasets_path / "test" / model_dataset_filename)
        datasets.append(test_dataset)
    except:
        raise Exception("load_datasets: Test dataset doesn't exist")
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
        + f"sparse_form={system_model_params.sparse_form}"
        + ".h5"
    )
    return suffix_filename
