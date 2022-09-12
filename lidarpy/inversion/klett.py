import numpy as np
import xarray as xr
from scipy.integrate import cumtrapz, trapz
from scipy.optimize import curve_fit
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular


def calib_strategy1(signal: np.array,
                    model: np.array,
                    reference: np.array):

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
    Input:

    z - Height Profile [m]
    pr - Lidar Profile [-]
    ref - Reference Height(s) [m]
    lambda_ - Lidar Wavelength [nm]
    lidar_ratio - Extinction to Backscatter Ratio [sr]
    p_air - Atmospheric pressure profile [Pa]
    t_air - Atmospheric temperature profile [K]
    co2ppmv - CO2 concentration [ppmv]


    Authors:
    Pablo Ristori    (pablo.ristori@gmail.com) CEILAP, UNIDEF (CITEDEF-CONICET), Argentina
    Lidia Otero      (lidia1116@gmail.com)     CEILAP, UNIDEF (CITEDEF-CONICET), Argentina
    """
    _alpha = dict()
    _alpha_std = None
    _beta = dict()
    _beta_std = None
    _lr = dict()
    fit_parameters = None
    _calib_strategies = {True: calib_strategy1, False: calib_strategy2}

    def __init__(self, lidar_data: xr.Dataset, wavelength: int, lidar_ratio: float, p_air: np.ndarray,
                 t_air: np.ndarray, z_ref: list, pc: bool = True, co2ppmv: int = 392, correct_noise: bool = True,
                 mc_iter=None):
        if wavelength in lidar_data.dims:
            self.signal = lidar_data.sel(wavelength=f"{wavelength}_{int(pc)}").data
        else:
            self.signal = lidar_data.data

        self.z = lidar_data.coords["altitude"].data
        self.mc_iter = mc_iter
        self.ref = lidar_data.coords["altitude"].sel(altitude=z_ref, method="nearest").data
        self._calib_strategy = self._calib_strategies[correct_noise]
        self._lr['aer'] = lidar_ratio

        self._get_alpha_beta_molecular(p_air, t_air, wavelength * 1e-9, co2ppmv)

    def __str__(self):
        return f"Lidar ratio = {self._lr['aer']}"

    def get_beta(self) -> dict:
        return self._beta.copy()

    def get_alpha(self) -> dict:
        return self._alpha.copy()

    def get_lidar_ratio(self) -> dict:
        return self._lr.copy()

    def set_lidar_ratio(self, lidar_ratio):
        self._lr['aer'] = lidar_ratio
        return self

    def get_beta_std(self) -> np.array:
        return self._beta_std.copy()

    def get_alpha_std(self) -> np.array:
        return self._alpha_std.copy()

    def get_model_mol(self) -> np.array:
        return self._beta['mol'] * np.exp(-2 * cumtrapz(self._alpha['mol'], self.z, initial=0)) / self.z ** 2

    def _get_alpha_beta_molecular(self, p_air, t_air, wavelength, co2ppmv):
        alpha_beta_mol = AlphaBetaMolecular(p_air, t_air, wavelength, co2ppmv)
        self._alpha['mol'], self._beta['mol'], self._lr['mol'] = alpha_beta_mol.get_params()

    def _calib(self, signal):
        ref = np.where((self.z == self.ref[0]) | (self.z == self.ref[1]))[0]

        if len(ref) > 1:
            signal, model, self.fit_parameters = self._calib_strategy(signal=signal.copy(),
                                                                      model=self.get_model_mol(),
                                                                      reference=np.arange(ref[0], ref[-1], dtype=int))
            beta_ref = self._beta['mol'][ref[0]] * signal[ref[0]] / model[ref[0]]
        else:
            signal = signal
            beta_ref = self._beta['mol'][ref[0]]

        return beta_ref, signal, ref[0]

    def fit(self):
        if self.mc_iter is not None:
            signals = np.random.poisson(self.signal, size=(self.mc_iter, len(self.signal)))

            betas_aer = []
            alphas_aer = []
            for signal in signals:
                beta_ref, signal, ref0 = self._calib(signal)

                corrected_signal = signal * self.z ** 2

                spp = corrected_signal * np.exp(- 2 * cumtrapz(x=self.z,
                                                               y=(self._lr['aer']
                                                                  - self._lr['mol']) * self._beta['mol'],
                                                               initial=0))

                sppr = spp / spp[ref0]

                betas_aer.append(sppr / (1 / beta_ref - (cumtrapz(x=self.z, y=2 * self._lr['aer'] * sppr, initial=0)
                                                         - trapz(x=self.z[:ref0],
                                                                 y=2 * self._lr['aer'] * sppr[:ref0])))
                                 - self._beta['mol'])

                alphas_aer.append(betas_aer[-1] * self._lr['aer'])
            self._alpha['aer'] = np.mean(alphas_aer, axis=0)
            self._alpha['tot'] = self._alpha['mol'] + self._alpha['aer']
            self._alpha_std = np.std(alphas_aer, ddof=1, axis=0)
            self._beta['aer'] = np.mean(betas_aer, axis=0)
            self._beta_std = np.std(betas_aer, ddof=1, axis=0)
            self._beta['tot'] = self._beta['aer'] + self._beta['mol']

            return (self._alpha["aer"].copy(),
                    self._alpha_std.copy(),
                    self._beta["aer"].copy(),
                    self._beta_std.copy(),
                    self._lr["aer"])

        beta_ref, signal, ref0 = self._calib(self.signal)

        corrected_signal = signal * self.z ** 2

        spp = corrected_signal * np.exp(- 2 * cumtrapz(x=self.z,
                                                       y=(self._lr['aer'] - self._lr['mol']) * self._beta['mol'],
                                                       initial=0))

        sppr = spp / spp[ref0]

        self._beta['tot'] = sppr / (1 / beta_ref - (cumtrapz(x=self.z, y=2 * self._lr['aer'] * sppr, initial=0)
                                                    - trapz(x=self.z[:ref0], y=2 * self._lr['aer'] * sppr[:ref0])))

        self._beta['aer'] = self._beta['tot'] - self._beta['mol']

        self._alpha['aer'] = self._beta['aer'] * self._lr['aer']

        self._alpha['tot'] = self._alpha['mol'] + self._alpha['aer']

        return self._alpha["aer"].copy(), self._beta["aer"].copy(), self._lr["aer"]
