import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import curve_fit
from lidarpy.utils.functions import z_finder


def calib_strategy1(signal: np.array,
                    model: np.array,
                    reference: np.array):
    """
    Calibrate a signal against a model using a linear strategy.

    Parameters:
    ----------
    signal : np.array
        Input signal to be calibrated.
    model : np.array
        Model against which the signal is calibrated.
    reference : np.array
        Indices considered for calibration.

    Returns:
    -------
    tuple
        A tuple containing the calibrated signal, model, and the coefficients of the linear fit.
    """
    reference -= 1

    if reference[-1] >= len(signal):
        reference = np.arange(reference[0], len(signal))

    mean_signal = np.mean(signal[reference])
    signal = signal / mean_signal

    mean_model = np.mean(model[reference])
    model = model / mean_model

    coef_slope_linear, *_ = curve_fit(f=lambda x, a, b: a * x + b, xdata=model[reference], ydata=signal[reference])

    signal = (signal - coef_slope_linear[1]) / coef_slope_linear[0]

    coef_slope_linear = [coef_slope_linear[0] * mean_signal / mean_model, coef_slope_linear[1] * mean_signal]

    return signal, model, coef_slope_linear


def calib_strategy2(signal: np.array,
                    model: np.array,
                    reference: np.array):
    """
    Calibrate a signal against a model using a simplified linear strategy.

    Parameters:
    ----------
    signal : np.array
        Input signal to be calibrated.
    model : np.array
        Model against which the signal is calibrated.
    reference : np.array
        Indices considered for calibration.

    Returns:
    -------
    tuple
        A tuple containing the calibrated signal, model, and the coefficient of the fit.
    """
    mean_signal = np.mean(signal[reference])
    signal = signal / mean_signal

    mean_model = np.mean(model[reference])
    model = model / mean_model

    coef_slope, *_ = curve_fit(f=lambda x, a: a * x, xdata=model[reference], ydata=signal[reference])

    signal = signal / coef_slope[0]

    coef_slope = [coef_slope[0] * mean_signal / mean_model]

    return signal, model, coef_slope


class Klett:
    """
    Implements the Klett inversion algorithm for lidar inversion.

    Attributes:
    ----------
    z : array
        Height profile in meters.
    pr : array
        Lidar profile.
    ref : array
        Reference height(s) in meters.
    lambda_ : float
        Lidar wavelength in nanometers.
    lidar_ratio : float
        Extinction to backscatter ratio in sr.
    p_air : array
        Atmospheric pressure profile in Pa.
    t_air : array
        Atmospheric temperature profile in K.
    co2ppmv : float
        CO2 concentration in ppmv.

    Methods:
    -------
    get_beta() -> dict:
        Returns aerosol backscatter coefficients.
    get_alpha() -> dict:
        Returns aerosol extinction coefficients.
    get_lidar_ratio() -> dict:
        Returns the lidar ratios.
    set_lidar_ratio(lidar_ratio: float):
        Sets a new lidar ratio.
    get_model_mol() -> np.array:
        Returns the molecular model for the lidar signal.
    fit() -> tuple:
        Fits the lidar inversion using the Klett inversion algorithm.

    Authors:
    - Pablo Ristori (pablo.ristori@gmail.com) CEILAP, UNIDEF (CITEDEF-CONICET), Argentina
    - Lidia Otero (lidia1116@gmail.com) CEILAP, UNIDEF (CITEDEF-CONICET), Argentina
    """
    _alpha = dict()
    _beta = dict()
    _lr = dict()
    tau = None
    tau_std = None
    mc_iter = None
    fit_parameters = None
    _calib_strategies = {True: calib_strategy1, False: calib_strategy2}

    def __init__(self, rangebin: np.array, signal: np.array, molecular_data: xr.Dataset, lidar_ratio: float,
                 molecular_reference_region: list, correct_noise: bool = True):
        self.signal = signal.copy()
        self.rangebin = rangebin.copy()
        self.ref = molecular_reference_region
        self._calib_strategy = self._calib_strategies[correct_noise]
        self._alpha['mol'], self._beta['mol'], self._lr['mol'] = (molecular_data.alpha.data, molecular_data.beta.data,
                                                                  molecular_data.lidar_ratio.data)
        self._lr['aer'] = lidar_ratio

    def __str__(self):
        return f"Lidar ratio = {self._lr['aer']}"

    def get_beta(self) -> dict:
        return self._beta.copy()

    def get_alpha(self) -> dict:
        return self._alpha.copy()

    def get_lidar_ratio(self) -> dict:
        return self._lr.copy()

    def set_lidar_ratio(self, lidar_ratio: float):
        self._lr['aer'] = lidar_ratio
        return self

    def set_correction(self, correct_noise: bool):
        self._calib_strategy = self._calib_strategies[correct_noise]
        return self

    def get_model_mol(self) -> np.array:
        return (self._beta['mol'] * np.exp(-2 * cumulative_trapezoid(self._alpha['mol'], self.rangebin, initial=0))
                / self.rangebin ** 2)

    def _calibration(self, signal):
        ref = z_finder(self.rangebin, self.ref)

        if len(ref) > 1:
            signal, model, self.fit_parameters = self._calib_strategy(signal=signal.copy(),
                                                                      model=self.get_model_mol(),
                                                                      reference=np.arange(*ref))

            beta_ref = self._beta['mol'][ref[0]] * signal[ref[0]] / model[ref[0]]
        else:
            signal = signal
            beta_ref = self._beta['mol'][ref[0]]

        return beta_ref, signal, ref[0]

    def fit(self):
        beta_ref, signal, ref0 = self._calibration(self.signal)

        corrected_signal = signal * self.rangebin ** 2

        spp = corrected_signal * np.exp(- 2 * cumulative_trapezoid(x=self.rangebin,
                                                       y=(self._lr['aer'] - self._lr['mol']) * self._beta['mol'],
                                                       initial=0))

        sppr = spp / spp[ref0]

        self._beta['tot'] = sppr / (
                1 / beta_ref - (cumulative_trapezoid(x=self.rangebin, y=2 * self._lr['aer'] * sppr, initial=0)
                                - trapezoid(x=self.rangebin[:ref0],
                                        y=2 * (self._lr['aer'] * np.ones_like(sppr))[:ref0] * sppr[:ref0])))

        self._beta['aer'] = self._beta['tot'] - self._beta['mol']

        self._alpha['aer'] = self._beta['aer'] * self._lr['aer']

        self._alpha['tot'] = self._alpha['mol'] + self._alpha['aer']

        self.tau = trapezoid(self._alpha["aer"], self.rangebin)

        return self._alpha["aer"].copy(), self._beta["aer"].copy(), self._lr["aer"]
