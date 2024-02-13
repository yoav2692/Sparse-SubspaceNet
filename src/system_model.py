"""Subspace-Net 
Details
----------
Name: system_model.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the SystemModel class for defining the settings of the DoA estimation system model.
"""

# Imports
import numpy as np
from src.classes import *
from src.sensors_arrays import *

class SystemModelParams:
    """Class for setting parameters of a system model."""

    def __init__(self):
        """
        Initialize the SystemModelParams object.

        Parameters:
            None

        Attributes:
            M (int): Number of sources.
            N (int): Number of sensors.
            T (int): Number of observations.
            signal_type (Signal_type): Signal type (Signal_type.narrowband.value or Signal_type.broadband.value).
            freq_values (list): Frequency values for broadband signal.
            signal_nature (Signal_nature): Signal nature (non_coherent or coherent).
            snr (float): Signal-to-noise ratio.
            eta (float): Sensor location deviation.
            sv_noise_var (float): Steering vector added noise variance.

        Returns:
            None
        """
        self.M = None  # Number of sources
        self.N = None  # Number of sensors
        self.T = None  # Number of observations
        self.signal_type = Signal_type.DEFAULT.value
        self.freq_values = [0, 500]  # Frequency values for broadband signal
        self.signal_nature = (
            Signal_nature.DEFAULT.value  # Signal nature (non_coherent or coherent)
        )
        self.snr = 10  # Signal-to-noise ratio
        self.eta = 0  # Sensor location deviation
        self.sv_noise_var = 0  # Steering vector added noise variance
        self.sensors_array_form = "ULA"
        self.doa_gap = 15
        self.doa_range = 90
        self.is_known_num_sources = True
        self.multi_num_sources_flag = False

    def set_multi_num_sources_flag(self, multi_num_sources_flag: bool):
        self.multi_num_sources_flag = multi_num_sources_flag
        return self
    
    def set_is_known_num_sources(self, is_known_num_sources: bool):
        self.is_known_num_sources = is_known_num_sources
        return self

    def set_num_sources(self, M: int):
        """
        Set the number of sources.

        Parameters:
            M (int): Number of sources.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        if self.multi_num_sources_flag:
            self.M = np.random.randint(M) + 1
        else:
            self.M = M
        return self

    def set_doa_gap(self, gap: int):
        """
        Set the gap between DoA's.

        Parameters:
            gap (int): sources minimal gap.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.doa_gap = gap
        return self

    def set_doa_range(self, range: float):
        """
        Set the DoA range.

        Parameters:
            range (float): maximal range of DoA's.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.doa_range = range
        return self

    def set_num_sensors(self, N: int):
        """
        Set the number of sensors.

        Parameters:
            N (int): Number of sensors.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.N = N
        return self

    def set_num_observations(self, T: int):
        """
        Set the number of observations.

        Parameters:
            T (int): Number of observations.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.T = T
        return self

    def set_signal_type(self, signal_type: str, freq_values: list = [0, 500]):
        """
        Set the signal type.

        Parameters:
            signal_type (str): Signal type (Signal_type.narrowband.value or Signal_type.broadband.value).
            freq_values (list, optional): Frequency values for broadband signal.
                Defaults to [0, 500].

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.signal_type = signal_type
        if signal_type.startswith(Signal_type.broadband.value):
            self.freq_values = freq_values
        return self

    def set_signal_nature(self, signal_nature: str):
        """
        Set the signal nature.

        Parameters:
            signal_nature (str): Signal nature (Signal_nature.non_coherent.value or Signal_nature.coherent.value).

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.signal_nature = signal_nature
        return self

    def set_snr(self, snr: float):
        """
        Set the signal-to-noise ratio.

        Parameters:
            snr (float): Signal-to-noise ratio.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.snr = snr
        return self

    def set_sensors_dev(self, eta: float):
        """
        Set the level of deviation from sensor location.

        Parameters:
            eta (float): Level of deviation from sensor location.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.eta = eta
        return self

    def set_sv_noise(self, sv_noise_var: float):
        """
        Set the steering vector added noise variance.

        Parameters:
            sv_noise_var (float): Steering vector added noise variance.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.sv_noise_var = sv_noise_var
        return self

    def set_sensors_array(self, sensors_array: SensorsArray):
        """
        Set the sparse formation of the sensors array.

        Parameters:
            sensors_array_form: the array formation.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.sensors_array = sensors_array
        self.sensors_array_form = sensors_array.sensors_array_form
        self.set_num_sensors(sensors_array.last_sensor_loc)
        return self


class SystemModel(object):
    def __init__(self, system_model_params: SystemModelParams):
        """Class used for defining the settings of the system model.

        Attributes:
        -----------
            signal_type (str): Signals type. Options: Signal_type.narrowband.value, Signal_type.broadband.value.
            N (int): Number of sensors.
            M (int): Number of sources.
            freq_values (list, optional): Frequency range for broadband signals. Defaults to None.
            min_freq (dict): Minimal frequency value for different scenarios.
            max_freq (dict): Maximal frequency value for different scenarios.
            f_rng (dict): Frequency range of interest for different scenarios.
            f_sampling (dict): Sampling rate for different scenarios.
            time_axis (dict): Time axis for different scenarios.
            dist (dict): Distance between array elements for different scenarios.
            array (np.ndarray): Array of sensor locations.

        Methods:
        --------
            define_scenario_params(freq_values: list): Defines the signal_type parameters.
            create_array(): Creates the array of sensor locations.
            steering_vec(theta: np.ndarray, f: float = 1, array_form: str = "ULA",
                eta: float = 0, geo_noise_var: float = 0) -> np.ndarray: Computes the steering vector.

        """
        self.params = system_model_params
        # Assign signal type parameters
        self.define_scenario_params()
        # Define array indices
        self.create_array()

    def define_scenario_params(self):
        """Defines the signal type parameters based on the specified frequency values."""
        freq_values = self.params.freq_values
        # Define minimal frequency value
        self.min_freq = {
            Signal_type.narrowband.value: None,
            Signal_type.broadband.value: freq_values[0],
        }
        # Define maximal frequency value
        self.max_freq = {
            Signal_type.narrowband.value: None,
            Signal_type.broadband.value: freq_values[1],
        }
        # Frequency range of interest
        self.f_rng = {
            Signal_type.narrowband.value: None,
            Signal_type.broadband.value: np.linspace(
                start=self.min_freq[Signal_type.broadband.value],
                stop=self.max_freq[Signal_type.broadband.value],
                num=self.max_freq[Signal_type.broadband.value]
                - self.min_freq[Signal_type.broadband.value],
                endpoint=False,
            ),
        }
        # Define sampling rate as twice the maximal frequency
        self.f_sampling = {
            Signal_type.narrowband.value: None,
            Signal_type.broadband.value: 2
            * (
                self.max_freq[Signal_type.broadband.value]
                - self.min_freq[Signal_type.broadband.value]
            ),
        }
        # Define time axis
        self.time_axis = {
            Signal_type.narrowband.value: None,
            Signal_type.broadband.value: np.linspace(
                0, 1, self.f_sampling[Signal_type.broadband.value], endpoint=False
            ),
        }
        # distance between array elements
        self.dist = {
            Signal_type.narrowband.value: 1 / 2,
            Signal_type.broadband.value: 1
            / (
                2
                * (
                    self.max_freq[Signal_type.broadband.value]
                    - self.min_freq[Signal_type.broadband.value]
                )
            ),
        }

    def create_array(self):
        """create an array of sensors locations"""
        self.array = np.linspace(0, self.params.N, self.params.N, endpoint=False)

    def steering_vec(self, theta: np.ndarray, f: float = 1, array_form="ULA"):
        """Computes the steering vector based on the specified parameters.

        Args:
        -----
            theta (np.ndarray): Array of angles.
            f (float, optional): Frequency. Defaults to 1.
            array_form (str, optional): Array form. Defaults to "ULA".

        Returns:
        --------
            np.ndarray: Computed steering vector.

        """
        sv_noise_var = self.params.sv_noise_var
        f_sv = {Signal_type.narrowband.value: 1, Signal_type.broadband.value: f}
        if array_form.startswith("ULA"):
            # define uniform deviation in spacing (for each sensor)
            mis_distance = np.random.uniform(
                low=-1 * self.params.eta, high=self.params.eta, size=self.params.N
            )
            # define noise added to steering vector
            mis_geometry_noise = np.sqrt(self.params.sv_noise_var) * (
                np.random.randn(self.params.N)
            )
            return (
                np.exp(
                    -2
                    * 1j
                    * np.pi
                    * f_sv[self.params.signal_type]
                    * (mis_distance + self.dist[self.params.signal_type])
                    * self.array
                    * np.sin(theta)
                )
                + mis_geometry_noise
            )
        else:
            raise Exception(
                f"SystemModel.steering_vec: array form {array_form} is not defined"
            )

    def __str__(self):
        """Returns a string representation of the SystemModel object.
        ...

        """
        print("System Model Summery:")
        for key, value in self.__dict__.items():
            print(key, " = ", value)
        return "End of Model"
