"""Subspace-Net 
Details
----------
Name: criterions.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 03/06/23

Purpose:
--------
The purpose of this script is to define and document several loss functions (RMSPELoss and MSPELoss)
and a helper function (permute_prediction) for calculating the Root Mean Square Periodic Error (RMSPE)
and Mean Square Periodic Error (MSPE) between predicted values and target values.
The script also includes a utility function RMSPE and MSPE that calculates the RMSPE and MSPE values
for numpy arrays.

This script includes the following Classes anf functions:

* permute_prediction: A function that generates all possible permutations of a given prediction tensor.
* RMSPELoss (class): A custom PyTorch loss function that calculates the RMSPE loss between predicted values
    and target values. It inherits from the nn.Module class and overrides the forward method to perform
    the loss computation.
* MSPELoss (class): A custom PyTorch loss function that calculates the MSPE loss between predicted values
  and target values. It inherits from the nn.Module class and overrides the forward method to perform the loss computation.
* RMSPE (function): A function that calculates the RMSPE value between the DOA predictions and target DOA values for numpy arrays.
* MSPE (function): A function that calculates the MSPE value between the DOA predictions and target DOA values for numpy arrays.
* set_criterions(function): Set the loss criteria based on the criterion name.

"""

import numpy as np
import torch.nn as nn
import torch
from src.classes import *
from src.utils import *
from itertools import permutations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");

def permute_prediction(prediction: torch.Tensor):
    """
    Generates all the available permutations of the given prediction tensor.

    Args:
        prediction (torch.Tensor): The input tensor for which permutations are generated.

    Returns:
        torch.Tensor: A tensor containing all the permutations of the input tensor.

    Examples:
        >>> prediction = torch.tensor([1, 2, 3])
        >>>> permute_prediction(prediction)
            torch.tensor([[1, 2, 3],
                          [1, 3, 2],
                          [2, 1, 3],
                          [2, 3, 1],
                          [3, 1, 2],
                          [3, 2, 1]])
        
    """
    torch_perm_list = []
    for p in list(permutations(range(prediction.shape[0]),prediction.shape[0])):
        torch_perm_list.append(prediction.index_select( 0, torch.tensor(list(p), dtype = torch.int64).to(device)))
    predictions = torch.stack(torch_perm_list, dim = 0)
    return predictions

class RMSPELoss(nn.Module):
    """Root Mean Square Periodic Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the RMSPE loss between the predictions and target values.

    Example:
        criterion = RMSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """
    def __init__(self , method : str = Loss_method.DEFAULT.value):
        super(RMSPELoss, self).__init__()
        self.method = method

    def forward(self, doa_predictions: torch.Tensor, doa: torch.Tensor):
        """Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSPE.
        The minimum RMSPE value among all permutations is selected for each sample.
        Finally, the method sums up the RMSPE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).

        Returns:
            torch.Tensor: The computed RMSPE loss.

        Raises:
            None
        """
        rmspe = []
        for iter in range(doa_predictions.shape[0]):
            rmspe_list = []
            batch_predictions = doa_predictions[iter].to(device)
            targets = doa[iter].to(device)
            if self.method.startswith(Loss_method.full_permute.value):
                prediction_perm = permute_prediction(batch_predictions).to(device)
                for prediction in prediction_perm:
                    # Calculate error with modulo pi
                    error = prediction - targets
                    if "periodic" in self.method:
                        error = pi_periodic(error)
                    # Calculate RMSE over all permutations
                    rmspe_val = (1 / np.sqrt(len(targets))) * torch.linalg.norm(error)
                    rmspe_list.append(rmspe_val)
                
                rmspe_tensor = torch.stack(rmspe_list, dim = 0)
                # Choose minimal error from all permutations
                rmspe_min = torch.min(rmspe_tensor)
            elif  self.method.startswith(Loss_method.no_permute.value):
                batch_predictions , indices = torch.sort(batch_predictions)
                # targets , indices = torch.sort(targets)
                # assuming targets are sorted and using L2 norm is used
                error = batch_predictions - targets
                if "periodic" in self.method:
                    error = pi_periodic(error)
                # Calculate RMSE over all permutations
                rmspe_val = (1 / np.sqrt(len(targets))) * torch.linalg.norm(error)
                rmspe_min = rmspe_val
            
            rmspe.append(rmspe_min)
        result = torch.sum(torch.stack(rmspe, dim = 0))
        return result

class RMSELoss(nn.Module):
    """Root Mean Square Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the RMSPE loss between the predictions and target values.

    Example:
        criterion = RMSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """
    def __init__(self , method : str = Loss_method.sort.value):
        super(RMSELoss, self).__init__()
        self.method = method

    def forward(self, doa_predictions: torch.Tensor, doa: torch.Tensor):
        """Compute the RMSE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        Finally, the method sums up the RMSE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).

        Returns:
            torch.Tensor: The computed RMSE loss.

        Raises:
            None
        """
        loss_per_iter = []
        for iter in range(doa_predictions.shape[0]):
            targets = doa[iter].to(device)
            batch_predictions = doa_predictions[iter].to(device)
            if Loss_method.sort.value in self.method:
                # assuming targets are sorted and using L2 norm is used
                batch_predictions , indices = torch.sort(batch_predictions)
                # targets , indices = torch.sort(targets)
            error = batch_predictions - targets
            #rmspe_val = RMSE()
            rmspe_val = (1 / np.sqrt(len(targets))) * torch.linalg.norm(error)
            loss_per_iter.append(rmspe_val)
        loss = torch.sum(torch.stack(loss_per_iter, dim = 0))
        return loss

def RMSPE(doa_predictions: np.ndarray, doa: np.ndarray):
    """
    Calculate the Root Mean Square Periodic Error (RMSPE) between the DOA predictions and target DOA values.

    Args:
        doa_predictions (np.ndarray): Array of DOA predictions.
        doa (np.ndarray): Array of target DOA values.

    Returns:
        float: The computed RMSPE value.

    Raises:
        None
    """
    doa = safe_np_array_cast(doa).squeeze()
    rmspe_list = []
    # Calculate error with modulo pi
    for p in list(permutations(doa_predictions, len(doa_predictions))):
        p = safe_np_array_cast(p)
        rmspe_list.append(RMS(pi_periodic(p - doa)))
    # Choose minimal error from all permutations
    return np.min(rmspe_list)

def RMSE(predictions: np.ndarray, targets: np.ndarray):
    targets = safe_np_array_cast(targets)
    return RMS(predictions - targets)

def RMS(err: np.ndarray):
    return np.linalg.norm(err) / np.sqrt(len(err))

def MSPE(doa_predictions: np.ndarray, doa: np.ndarray):
    """Calculate the Mean Square Percentage Error (RMSPE) between the DOA predictions and target DOA values.

    Args:
        doa_predictions (np.ndarray): Array of DOA predictions.
        doa (np.ndarray): Array of target DOA values.

    Returns:
        float: The computed RMSPE value.

    Raises:
        None
    """
    rmspe_list = []
    for p in list(permutations(doa_predictions, len(doa_predictions))):
        p = safe_np_array_cast(p)
        doa = safe_np_array_cast(doa)
        # Calculate error with modulo pi
        error = pi_periodic(p - doa)
        # Calculate MSE over all permutations
        rmspe_val = (1 / len(p)) * (np.linalg.norm(error) ** 2)
        rmspe_list.append(rmspe_val)
    # Choose minimal error from all permutations
    return np.min(rmspe_list)

def set_criterions(criterion_name:str):
    """
    Set the loss criteria based on the criterion name.

    Parameters:
        criterion_name (str): Name of the criterion.

    Returns:
        criterion (nn.Module): Loss criterion for model evaluation.
        subspace_criterion (Callable): Loss criterion for subspace method evaluation.

    Raises:
        Exception: If the criterion name is not defined.
    """
    if criterion_name.startswith(Criterion.RMSPE.value):
        criterion = RMSPELoss()
        subspace_criterion = RMSPE
    elif criterion_name.startswith(Criterion.RMSE.value):
        criterion = RMSELoss()
        subspace_criterion = RMSE
    else:
        raise Exception(f"criterions.set_criterions: Criterion {criterion_name} is not defined")
    print(f"Loss measure = {criterion_name}")
    return criterion, subspace_criterion

if __name__ == "__main__":
    prediction = torch.tensor([1, 2, 3])
    print(permute_prediction(prediction))