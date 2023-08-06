"""Subspace-Net 
Details
----------
Name: signal_creation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the Samples class, which inherits from SystemModel class.
This class is used for defining the samples model.
"""

# Imports
import numpy as np
from src.classes import *
from src.system_model import SystemModel, SystemModelParams
from src.sensors_arrays import *
from src.utils import D2R


class Samples(SystemModel):
    """
    Class used for defining and creating signals and observations.
    Inherits from SystemModel class.

    ...

    Attributes:
    -----------
        doa (np.ndarray): Array of angels (directions) of arrival.

    Methods:
    --------
        set_doa(doa): Sets the direction of arrival (DOA) for the signals.
        samples_creation(noise_mean: float = 0, noise_variance: float = 1, signal_mean: float = 0,
            signal_variance: float = 1): Creates samples based on the specified mode and parameters.
        noise_creation(noise_mean, noise_variance): Creates noise based on the specified mean and variance.
        signal_creation(signal_mean=0, signal_variance=1, SNR=10): Creates signals based on the specified mode and parameters.
    """

    def __init__(self, system_model_params: SystemModelParams):
        """Initializes a Samples object.

        Args:
        -----
        system_model_params (SystemModelParams): an instance of SystemModelParams,
            containing all relevant system model parameters.

        """
        super().__init__(system_model_params)

    def set_doa(self, doa):
        """
        Sets the direction of arrival (DOA) for the signals.

        Args:
        -----
            doa (np.ndarray): Array containing the DOA values.

        """

        def create_doa_with_gap(gap: float):
            """Create angles with a value gap.

            Args:
            -----
                gap (float): Minimal gap value.

            Returns:
            --------
                np.ndarray: DOA array.

            """
            M = self.params.M
            while True:
                DOA = np.round(np.random.rand(M) * 180, decimals=2) - 90
                DOA.sort()
                diff_angles = np.array(
                    [np.abs(DOA[i + 1] - DOA[i]) for i in range(M - 1)]
                )
                if (np.sum(diff_angles > gap) == M - 1) and (
                    np.sum(diff_angles < (180 - gap)) == M - 1
                ):
                    break
            return DOA

        if doa == None:
            # Generate angels with gap greater than 0.2 rad (nominal case)
            self.doa = np.array(create_doa_with_gap(gap=15)) * D2R
        else:
            # Generate
            self.doa = np.array(doa) * D2R
        self.doa = np.sort(self.doa)

    def samples_creation(
        self,
        noise_mean: float = 0,
        noise_variance: float = 1,
        signal_mean: float = 0,
        signal_variance: float = 1,
    ):
        """Creates samples based on the specified mode and parameters.

        Args:
        -----
            noise_mean (float, optional): Mean of the noise. Defaults to 0.
            noise_variance (float, optional): Variance of the noise. Defaults to 1.
            signal_mean (float, optional): Mean of the signal. Defaults to 0.
            signal_variance (float, optional): Variance of the signal. Defaults to 1.

        Returns:
        --------
            tuple: Tuple containing the created samples, signal, steering vectors, and noise.

        Raises:
        -------
            Exception: If the signal_type is not defined.

        """
        # Generate signal matrix
        signal = self.signal_creation(signal_mean, signal_variance)
        # Generate noise matrix
        noise = self.noise_creation(noise_mean, noise_variance)
        # Generate Narrowband samples
        if self.params.signal_type.startswith(Signal_type.narrowband.value):
            A = np.array([self.steering_vec(theta) for theta in self.doa]).T
            samples = (A @ signal) + noise
            if not self.params.sensors_array_form.startswith("ULA"):
                samples = self.sample_missing_sensors_handle(samples)
            return samples, signal, A, noise
        # Generate broadband samples
        elif self.params.signal_type.startswith(Signal_type.broadband.value):
            samples = []
            SV = []

            for idx in range(self.f_sampling[Signal_type.broadband.value]):
                # mapping from index i to frequency f
                if idx > int(self.f_sampling[Signal_type.broadband.value]) // 2:
                    f = -int(self.f_sampling[Signal_type.broadband.value]) + idx
                else:
                    f = idx
                A = np.array([self.steering_vec(theta, f) for theta in self.doa]).T
                samples.append((A @ signal[:, idx]) + noise[:, idx])
                SV.append(A)
            samples = np.array(samples)
            SV = np.array(SV)
            samples_time_domain = np.fft.ifft(samples.T, axis=1)[:, : self.params.T]
            return samples_time_domain, signal, SV, noise
        else:
            raise Exception(
                f"Samples.samples_creation: signal type {self.params.signal_type} is not defined"
            )
    def sample_missing_sensors_handle(self,samples):
        missing_sensors = [x for x in range(self.params.sensors_array.last_sensor_loc) if x not in self.params.sensors_array.locs] 
        if self.params.sensors_array.missing_sensors_handle_method == Missing_senors_handle_method.zeros.value:
            for missing_sensor in missing_sensors:
                samples[:,][missing_sensor] = 0
        elif self.params.sensors_array.missing_sensors_handle_method == Missing_senors_handle_method.phase_continuetion.value:
            for missing_sensor in missing_sensors:
                diffs = self.params.sensors_array.locs - missing_sensor
                phase_diff = diffs[np.argmin(abs(diffs))]
                closest_sensor = self.params.sensors_array.locs[np.argmin(abs(diffs))]
                # * f_sv[self.params.signal_type]
                samples[:,][missing_sensor] = samples[:,][closest_sensor] * np.exp(-1j* np.pi*phase_diff)
        return samples
        
    def noise_creation(self, noise_mean, noise_variance):
        """Creates noise based on the specified mean and variance.

        Args:
        -----
            noise_mean (float): Mean of the noise.
            noise_variance (float): Variance of the noise.

        Returns:
        --------
            np.ndarray: Generated noise.

        """
        # for narrowband signal_type Noise represented in the time domain
        if self.params.signal_type.startswith(Signal_type.narrowband.value):
            return (
                np.sqrt(noise_variance)
                * (np.sqrt(2) / 2)
                * (
                    np.random.randn(self.params.N, self.params.T)
                    + 1j * np.random.randn(self.params.N, self.params.T)
                )
                + noise_mean
            )
        # for broadband signal_type Noise represented in the frequency domain
        elif self.params.signal_type.startswith(Signal_type.broadband.value):
            noise = (
                np.sqrt(noise_variance)
                * (np.sqrt(2) / 2)
                * (
                    np.random.randn(self.params.N, len(self.time_axis[Signal_type.broadband.value]))
                    + 1j
                    * np.random.randn(self.params.N, len(self.time_axis[Signal_type.broadband.value]))
                )
                + noise_mean
            )
            return np.fft.fft(noise)
        else:
            raise Exception(
                f"Samples.noise_creation: signal type {self.params.signal_type} is not defined"
            )

    def signal_creation(self, signal_mean: float = 0, signal_variance: float = 1):
        """
        Creates signals based on the specified signal nature and parameters.

        Args:
        -----
            signal_mean (float, optional): Mean of the signal. Defaults to 0.
            signal_variance (float, optional): Variance of the signal. Defaults to 1.

        Returns:
        --------
            np.ndarray: Created signals.

        Raises:
        -------
            Exception: If the signal type is not defined.
            Exception: If the signal nature is not defined.
        """
        amplitude = 10 ** (self.params.snr / 10)
        # narrowband signal creation
        if self.params.signal_type == Signal_type.narrowband.value:
            if self.params.signal_nature == Signal_nature.non_coherent.value:
                # create M non-coherent signals
                return (
                    amplitude
                    * (np.sqrt(2) / 2)
                    * np.sqrt(signal_variance)
                    * (
                        np.random.randn(self.params.M, self.params.T)
                        + 1j * np.random.randn(self.params.M, self.params.T)
                    )
                    + signal_mean
                )

            elif self.params.signal_nature == Signal_nature.coherent.value:
                # Coherent signals: same amplitude and phase for all signals
                sig = (
                    amplitude
                    * (np.sqrt(2) / 2)
                    * np.sqrt(signal_variance)
                    * (
                        np.random.randn(1, self.params.T)
                        + 1j * np.random.randn(1, self.params.T)
                    )
                    + signal_mean
                )
                return np.repeat(sig, self.params.M, axis=0)

        # OFDM broadband signal creation
        elif self.params.signal_type.startswith(Signal_type.broadband.value):
            num_sub_carriers = self.max_freq[
                Signal_type.broadband.value
            ]  # number of subcarriers per signal
            if self.params.signal_nature == Signal_nature.non_coherent.value:
                # create M non-coherent signals
                signal = np.zeros(
                    (self.params.M, len(self.time_axis[Signal_type.broadband.value]))
                ) + 1j * np.zeros((self.params.M, len(self.time_axis[Signal_type.broadband.value])))
                for i in range(self.params.M):
                    for j in range(num_sub_carriers):
                        sig_amp = (
                            amplitude
                            * (np.sqrt(2) / 2)
                            * (np.random.randn(1) + 1j * np.random.randn(1))
                        )
                        signal[i] += sig_amp * np.exp(
                            1j
                            * 2
                            * np.pi
                            * j
                            * len(self.f_rng[Signal_type.broadband.value])
                            * self.time_axis[Signal_type.broadband.value]
                            / num_sub_carriers
                        )
                    signal[i] *= 1 / num_sub_carriers
                return np.fft.fft(signal)
            # Coherent signals: same amplitude and phase for all signals
            elif self.params.signal_nature == Signal_nature.coherent.value:
                signal = np.zeros(
                    (1, len(self.time_axis[Signal_type.broadband.value]))
                ) + 1j * np.zeros((1, len(self.time_axis[Signal_type.broadband.value])))
                for j in range(num_sub_carriers):
                    sig_amp = (
                        amplitude
                        * (np.sqrt(2) / 2)
                        * (np.random.randn(1) + 1j * np.random.randn(1))
                    )
                    signal += sig_amp * np.exp(
                        1j
                        * 2
                        * np.pi
                        * j
                        * len(self.f_rng[Signal_type.broadband.value])
                        * self.time_axis[Signal_type.broadband.value]
                        / num_sub_carriers
                    )
                signal *= 1 / num_sub_carriers
                return np.tile(np.fft.fft(signal), (self.params.M, 1))
            else:
                raise Exception(
                    f"signal nature {self.params.signal_nature} is not defined"
                )

        else:
            raise Exception(f"signal type {self.params.signal_type} is not defined")
