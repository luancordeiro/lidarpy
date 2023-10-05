import xarray as xr
import numpy as np
from lidarpy.utils.functions import molecular_model, z_finder
from lidarpy.inversion import Klett
from scipy.integrate import trapz
from scipy.interpolate import interp1d


class GetCod:
    """
    Implements a method for determining Cloud Optical Depth (COD) using lidar inversion.

    This class applies techniques described in molecular scattering and radiative transfer theory to calculate the cloud optical depth.
    The primary methods allow for both standard fitting and Monte Carlo techniques for uncertainty estimation.

    Attributes:
    - lidar_data (xr.Dataset): The dataset containing the lidar signals.
    - rangebin (array-like): Range or altitude bins of the lidar inversion.
    - wavelength (int): Emission wavelength of the lidar in nanometers.
    - molecular_data (xr.Dataset): The dataset containing the molecular or Rayleigh scattered signals.
    - pc (bool): Flag indicating if polarization correction should be applied.
    - co2ppmv (int): Concentration of CO2 in the atmosphere in ppmv.
    - fit_ref (list): Range indices for fitting the molecular profile.
    - transmittance_ref (int): Index for the reference altitude for determining transmittance.
    - mc_iter (int or None): Number of Monte Carlo iterations for uncertainty estimation.
    - indx (int): Index for a specific profile or inversion subset.
    - fit_ (bool): If True, it will fit the inversion.
    - delta_fit (int or None): Height interval for transmission fitting.
    - mc_niter (int): Number of monte carlo interations

    Methods:
    - fit: Calculates the cloud optical depth and the associated uncertainties using either the standard or Monte Carlo method.
    """

    _fit_delta_z: float = 3000
    _delta_z: float = 100
    _transmittance_bins: int = 150

    def __init__(self, signal: np.array, rangebin: np.array, sigma: np.array, molecular_data: xr.Dataset,
                 cloud_lims: list):
        self.signal = signal.copy()
        self.rangebin = rangebin.copy()
        self.sigma = sigma.copy()
        self.molecular_data = molecular_data
        self.cloud_lims = cloud_lims

    def set_fit_delta_z(self, fit_delta_z):
        self._fit_delta_z: float = fit_delta_z
        return self

    def set_delta_z(self, delta_z):
        self._delta_z: float = delta_z
        return self

    def set_transmittance_bins(self, transmittance_bins):
        self._transmittance_bins: float = transmittance_bins
        return self

    def fit(self):
        molecular_signal = molecular_model(self.signal,
                                           self.rangebin,
                                           self.sigma,
                                           self.molecular_data,
                                           [self.cloud_lims[0] - self._fit_delta_z - 50, self.cloud_lims[0] - 50],
                                           change_reference=None)

        molecular_rcs = molecular_signal * self.rangebin ** 2

        rcs = self.signal * self.rangebin ** 2

        transmittance_ref = z_finder(self.rangebin, self.cloud_lims[1] + self._delta_z)
        transmittance_ = (rcs[transmittance_ref: transmittance_ref + self._transmittance_bins]
                          / molecular_rcs[transmittance_ref: transmittance_ref + self._transmittance_bins])

        mean_ = transmittance_.mean()
        std_ = transmittance_.std(ddof=1)
        std_mean = std_ / np.sqrt(len(transmittance_))

        cod_mean = -0.5 * np.log(mean_)
        cod_std = 0.5 * std_mean / mean_

        return cod_mean, cod_std


class LidarRatioCalculator:
    def __init__(self, signal: np.array, rangebin: np.array, sigma: np.array, molecular_data: xr.Dataset, z_ref: list,
                 cloud_lims: list, mc=False, mc_niter=100, correct_noise: bool = True):
        self.signal = signal
        self.rangebin = rangebin
        self.sigma = sigma
        self.cloud_lims = cloud_lims
        self.molecular_data = molecular_data
        self.z_ref = z_ref
        self.correct_noise = correct_noise
        self.mc = mc
        self.mc_niter = mc_niter
        self.tau_mean, self.tau_std_mean = None, None
        self.tau_transmittance = None

    def _get_lr(self, tau_transmittance):
        lidar_ratios = np.arange(2, 50, 1)

        cloud_ind = z_finder(self.rangebin, self.cloud_lims)

        klett = Klett(signal=self.signal, rangebin=self.rangebin, molecular_data=self.molecular_data,
                      z_ref=self.z_ref,
                      lidar_ratio=1, correct_noise=self.correct_noise)

        taus = []
        for lidar_ratio in lidar_ratios:
            klett.set_lidar_ratio(lidar_ratio)
            alpha, *_ = klett.fit()
            taus.append(trapz(y=alpha[cloud_ind[0]:cloud_ind[1] + 1],
                              dx=self.rangebin[1] - self.rangebin[0]))

        difference = (np.array(taus) - tau_transmittance) ** 2

        f_diff = interp1d(lidar_ratios, difference, kind="cubic", fill_value="extrapolate")

        new_lr = np.linspace(8, 50, 100)

        new_diff = f_diff(new_lr)

        return new_lr[new_diff.argmin()]

    def _compute_tau(self):
        self.tau_mean, self.tau_std_mean = GetCod(signal=self.signal, rangebin=self.rangebin,
                                                  sigma=self.sigma, molecular_data=self.molecular_data,
                                                  cloud_lims=self.cloud_lims).fit()

        self.tau_transmittance = self.tau_mean + self.tau_std_mean * np.random.randn(self.mc_niter)

    def _has_valid_transmittance(self):
        valid_ratio = len(self.tau_transmittance[self.tau_transmittance >= 0]) / len(self.tau_transmittance)
        return valid_ratio > 0.5

    def _compute_valid_lrs(self):
        return [self._get_lr(tau_transmittance=tau) for tau in self.tau_transmittance if tau >= 0]

    @staticmethod
    def _get_lidar_ratio_values(lrs):
        valid_lrs = [lr for lr in lrs if 8.38 <= lr < 48]
        if len(valid_lrs) / len(lrs) >= 0.6:
            return np.mean(valid_lrs), np.std(valid_lrs, ddof=1)
        return 25, 0

    def fit(self):
        if self.tau_transmittance is None:
            self._compute_tau()

        if not self._has_valid_transmittance():
            return 25, 0, self.tau_mean, self.tau_std_mean

        if self.mc:
            lrs = self._compute_valid_lrs()
            lr, lr_error = self._get_lidar_ratio_values(lrs)
        else:
            lr = self._get_lr(tau_transmittance=self.tau_mean)
            lr_error = 0

        return lr, lr_error, self.tau_mean, self.tau_std_mean
