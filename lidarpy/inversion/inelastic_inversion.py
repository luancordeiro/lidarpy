import xarray as xr
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from lidarpy.utils.functions import z_finder
from lidarpy.utils.raman_functions import get_savgol_filter, beta_smooth


class Raman:
    """
    Implement the Raman inversion algorithm for lidar inversion.

    The Raman inversion algorithm extracts aerosol extinction and backscatter profiles
    from lidar inversion based on differences between elastic and inelastic (Raman) scattering signals.

    Attributes:
    ----------
    z : array
        Altitude in meters.
    elastic_signal : array
        Lidar signal from elastic scattering.
    inelastic_signal : array
        Lidar signal from inelastic (Raman) scattering.
    lidar_wavelength : float
        Wavelength of emitted lidar pulse in nanometers.
    raman_wavelength : float
        Wavelength of inelastically backscattered photons (from Raman scattering) in nanometers.
    angstrom_coeff : float
        Ångström exponent for wavelength dependence of aerosol scattering.
    p_air : array
        Atmospheric pressure profile in pascals.
    t_air : array
        Atmospheric temperature profile in kelvins.
    co2ppmv : float
        Atmospheric CO2 concentration in ppmv.
    diff_strategy : function
        Method for differentiation in aerosol extinction calculations.

    Methods:
    -------
    get_alpha() -> dict:
        Return aerosol extinction coefficients.
    get_beta() -> dict:
        Return aerosol backscatter coefficients.
    get_lidar_ratio() -> dict:
        Return lidar ratios (extinction to backscatter ratio).
    set_diff_strategy(diff_strategy=function) -> self
        Changes the differentiation strategy
    fit() -> tuple:
        Fit lidar inversion using Raman inversion algorithm.
    """
    _alpha = dict()
    _alpha_std = None
    _beta = dict()
    _beta_std = None
    _lr = dict()
    _mc_bool = True
    diff_values = None

    def __init__(self, rangebin: np.array, elastic_signal: np.array, inelastic_signal: np.array,
                 elastic_sigma: np.array, inelastic_sigma: np.array, elastic_molecular_data: xr.Dataset,
                 inelastic_molecular_data: xr.Dataset, raman_scatterer_numerical_density: np.array,
                 lidar_wavelength: int, raman_wavelength: int, angstrom_coeff: float, molecular_reference_region: list):
        self.elastic_signal = elastic_signal
        self.inelastic_signal = inelastic_signal
        self.elastic_uncertainty = elastic_sigma
        self.inelastic_uncertainty = inelastic_sigma
        self.rangebin = rangebin
        self.raman_scatterer_numerical_density = raman_scatterer_numerical_density
        self.lidar_wavelength = lidar_wavelength * 1e-9
        self.raman_wavelength = raman_wavelength * 1e-9
        self.angstrom_coeff = angstrom_coeff
        self._ref = z_finder(self.rangebin, molecular_reference_region)
        self._mean_ref = (self._ref[0] + self._ref[1]) // 2
        self._alpha['elastic_mol'] = elastic_molecular_data.alpha.data
        self._beta['elastic_mol'] = elastic_molecular_data.beta.data
        self._lr['elastic_mol'] = elastic_molecular_data.lidar_ratio.data
        self._alpha['inelastic_mol'] = inelastic_molecular_data.alpha.data
        self._beta['inelastic_mol'] = inelastic_molecular_data.beta.data
        self._lr['inelastic_mol'] = inelastic_molecular_data.lidar_ratio.data
        self._beta_smooth = beta_smooth
        self._diff_strategy = get_savgol_filter(21, 2)

    def get_alpha(self):
        return self._alpha.copy()

    def get_beta(self):
        return self._beta.copy()

    def get_lidar_ratio(self):
        return self._lr['aer'].copy()

    def set_diff_values(self, diff):
        self.diff_values = diff
        return self

    def set_beta_smooth(self, smoother):
        self._beta_smooth = smoother
        return self

    def _diff(self) -> np.array:
        if self.diff_values is None:
            return self._diff_strategy(self)
        return self.diff_values

    def _alpha_elastic_aer(self) -> np.array:
        diff_num_signal = self._diff()

        alpha = ((diff_num_signal - self._alpha['elastic_mol'] - self._alpha['inelastic_mol'])
                 / (1 + (self.lidar_wavelength / self.raman_wavelength) ** self.angstrom_coeff))

        return alpha

    def _alpha_elastic_total(self) -> np.array:
        return self._alpha["elastic_aer"] + self._alpha["elastic_mol"]

    def _alpha_inelastic_total(self) -> np.array:
        return self._alpha["inelastic_aer"] + self._alpha["inelastic_mol"]

    def _ref_value(self, y):
        p = np.poly1d(np.polyfit(self.rangebin[self._ref[0]: self._ref[1] + 1],
                                 y[self._ref[0]: self._ref[1] + 1], 1))

        return p(self.rangebin[self._mean_ref])

    def _beta_elastic_total(self) -> np.array:
        signal_ratio = ((self._ref_value(self.inelastic_signal) * self.elastic_signal
                         / (self._ref_value(self.elastic_signal) * self.inelastic_signal))
                        * (self.raman_scatterer_numerical_density
                           / self._ref_value(self.raman_scatterer_numerical_density)))

        attenuation_ratio = (np.exp(-cumulative_trapezoid(x=self.rangebin, y=self._alpha_inelastic_total(), initial=0)
                                    + trapezoid(x=self.rangebin[:self._mean_ref + 1],
                                            y=self._alpha_inelastic_total()[:self._mean_ref + 1]))
                             / np.exp(-cumulative_trapezoid(x=self.rangebin, y=self._alpha_elastic_total(), initial=0)
                                      + trapezoid(x=self.rangebin[:self._mean_ref + 1],
                                              y=self._alpha_elastic_total()[:self._mean_ref + 1])))

        beta_ref = self._ref_value(self._beta["elastic_mol"])

        return beta_ref * signal_ratio * attenuation_ratio

    def set_diff_strategy(self, diff_strategy):
        self._diff_strategy = diff_strategy
        return self

    def fit(self):
        self._alpha["elastic_aer"] = self._alpha_elastic_aer()

        self._alpha["inelastic_aer"] = (self._alpha["elastic_aer"]
                                        / (self.raman_wavelength / self.lidar_wavelength) ** self.angstrom_coeff)

        self._beta["elastic_aer"] = self._beta_elastic_total() - self._beta["elastic_mol"]

        self._lr['aer'] = self._alpha["elastic_aer"] / self._beta_smooth(self)

        return self._alpha["elastic_aer"].copy(), self._beta["elastic_aer"].copy(), self._lr['aer'].copy()
