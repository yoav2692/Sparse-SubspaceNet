"""Subspace-Net 
Details
----------
    Name: methods.py
    Authors: Y. Amiel
    Created: 02/09/23
    Edited: 02/09/23

Description:
-----------
This script contains the implementation of correlation calculation:

Purpose:
--------


Classes:
--------


Methods:
--------

"""

# Imports
import numpy as np
import scipy
import torch
# import cvxpy as cp
from src.models import SubspaceNet
from src.classes import *
from src.utils import *
from src.system_model import SystemModel
from src.utils import sum_of_diag, find_roots, R2D

def spatial_smoothing_covariance(X: np.ndarray, N: int):
    """
    Calculates the covariance matrix using spatial smoothing technique.

    Args:
    -----
        X (np.ndarray): Input samples matrix.

    Returns:
    --------
        covariance_mat (np.ndarray): Covariance matrix.
    """
    # Define the sub-arrays size
    sub_array_size = int(N / 2) + 1
    # Define the number of sub-arrays
    number_of_sub_arrays = N - sub_array_size + 1
    # Initialize covariance matrix
    covariance_mat = np.zeros((sub_array_size, sub_array_size)) + 1j * np.zeros(
        (sub_array_size, sub_array_size)
    )
    for j in range(number_of_sub_arrays):
        # Run over all sub-arrays
        x_sub = X[j : j + sub_array_size, :]
        # Calculate sample covariance matrix for each sub-array
        sub_covariance = np.cov(x_sub)
        # Aggregate sub-arrays covariances
        covariance_mat += sub_covariance
    # Divide overall matrix by the number of sources
    covariance_mat /= number_of_sub_arrays
    return covariance_mat

def spatial_stationary_covariance(X,array_locations):
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
    cov_matrix = np.cov(X[array_locations])
    virtual_size = array_locations[-1] + 1
    virtual_cov_matrix = np.zeros((virtual_size,virtual_size), dtype=complex)
    naive_cov_matrix_val = np.zeros(virtual_size, dtype=complex)
    naive_cov_matrix_elements = np.zeros(virtual_size)
    # other diag
    for array_loc_low_ind , array_loc_low in enumerate(array_locations):
        for array_loc_high_ind , array_loc_high in enumerate(array_locations):
            if array_loc_low_ind < array_loc_high_ind:
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
            loc = next(i for i,x in enumerate(array_locations) if x == m)# array_locations.index(m)
            virtual_cov_matrix[m,m] = cov_matrix[loc,loc]
        else:
            virtual_cov_matrix[m,m] =  naive_cov_matrix_val[0]
    return virtual_cov_matrix

def low_rank_matrix_complition(X,array_locations):
    cov_matrix = np.cov(X[array_locations])
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

def subspacnet_covariance(X, subspacenet_model: SubspaceNet):
    """
    Calculates the covariance matrix using the SubspaceNet model.

    Args:
    -----
        X (np.ndarray): Input samples matrix.
        subspacenet_model (SubspaceNet): Model used for covariance calculation.

    Returns:
    --------
        covariance_mat (np.ndarray): Covariance matrix.

    Raises:
    -------
        Exception: If the given model for covariance calculation is not from SubspaceNet type.
    """
    # Predict the covariance matrix using the SubspaceNet model
    subspacenet_model.eval()
    covariance_mat = subspacenet_model(X)[-1]
    # Convert to np.array type
    covariance_mat = safe_np_array_cast(covariance_mat).squeeze()
    return covariance_mat


def calculate_covariance(
    self, X: np.ndarray, mode: str = "sample", model: SubspaceNet = None
):
    """
    Calculates the covariance matrix based on the specified mode.

    Args:
    ----- 
        X (np.ndarray): Input samples matrix.
        mode (str): Covariance calculation mode. Options: "spatial_smoothing", Model_type.SubspaceNet.value, "sample".
        model: Optional model used for SubspaceNet covariance calculation.

    Returns:
    --------
        covariance_mat (np.ndarray): Covariance matrix.

    Raises:
    -------
        Exception: If the given model for covariance calculation is not from SubspaceNet type.
        Exception: If the covariance calculation mode is not defined.
    """

    if mode.startswith(cov_calc_method.spatial_stationary.value):
        return spatial_stationary_covariance(X,self.system_model.params.sensors_array.locs)
    elif mode.startswith(cov_calc_method.low_rank.value):
        return low_rank_matrix_complition(X,self.system_model.params.sensors_array.locs)
    elif mode.startswith(cov_calc_method.spatial_smoothing.value):
        return spatial_smoothing_covariance(X)
    elif mode.startswith(Model_type.SubspaceNet.value):
        return subspacnet_covariance(X, model)
    elif mode.startswith(cov_calc_method.DEFAULT.value):
        return np.cov(X)
    else:
        raise Exception(
            (
                f"SubspaceMethod.subspacnet_covariance: {mode} type for covariance\
            calculation is not defined"
            )
        )

# def autocorrelation_matrix(X: torch.Tensor, lag: int) -> torch.Tensor:
def autocorrelation_matrix(X: torch.Tensor, lag: int):
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
            
    Rx_lag = Rx_lag / (X.shape[-1] - lag)
    Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), 0)
    return Rx_lag


# def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
def create_autocorrelation_tensor(X: torch.Tensor, tau: int):
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
    if len(X.shape) == 3 and X.shape[0] == 1:
        X = X.squeeze()
    Rx_tau = []
    for i in range(tau):
        Rx_tau.append(autocorrelation_matrix(X, lag=i))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr

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
