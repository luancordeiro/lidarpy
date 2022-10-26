import numpy as np
import xarray as xr
from scipy.integrate import cumtrapz, trapz
from scipy.optimize import curve_fit
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular
from lidarpy.data.manipulation import z_finder, filter_wavelength


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
    tau = None
    tau_std = None
    mc_iter = None
    fit_parameters = None
    _calib_strategies = {True: calib_strategy1, False: calib_strategy2}
    _mc_bool = True

    def __init__(self, lidar_data: xr.Dataset, wavelength: int, p_air: np.ndarray, t_air: np.ndarray, z_ref: list,
                 lidar_ratio: float, pc: bool = True, co2ppmv: int = 392, correct_noise: bool = True,
                 mc_iter: int = None, tau_lims: np.array = None):
        if (mc_iter is not None) & (tau_lims is None):
            raise Exception("Para realizar mc, é necessário add mc_iter e tau_ind")
        self.signal = filter_wavelength(lidar_data, wavelength, pc)
        self.uncertainty = (filter_wavelength(lidar_data, wavelength, pc, "sigma")
                            if "sigma" in lidar_data.variables else None)
        self.rangebin = lidar_data.coords["rangebin"].data
        self.ref = z_ref
        self._calib_strategy = self._calib_strategies[correct_noise]
        self._get_alpha_beta_molecular(p_air, t_air, wavelength * 1e-9, co2ppmv)
        self._lr['aer'] = lidar_ratio
        self.mc_iter = mc_iter
        self.tau_lims = z_finder(self.rangebin, tau_lims)

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
        return (self._beta['mol'] * np.exp(-2 * cumtrapz(self._alpha['mol'], self.rangebin, initial=0))
                / self.rangebin ** 2)

    def _get_alpha_beta_molecular(self, p_air, t_air, wavelength, co2ppmv):
        alpha_beta_mol = AlphaBetaMolecular(p_air, t_air, wavelength, co2ppmv)
        self._alpha['mol'], self._beta['mol'], self._lr['mol'] = alpha_beta_mol.get_params()

    def _calib(self, signal):
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
        if (self.mc_iter is not None) & self._mc_bool:
            return self._mc_fit()

        beta_ref, signal, ref0 = self._calib(self.signal)

        corrected_signal = signal * self.rangebin ** 2

        spp = corrected_signal * np.exp(- 2 * cumtrapz(x=self.rangebin,
                                                       y=(self._lr['aer'] - self._lr['mol']) * self._beta['mol'],
                                                       initial=0))

        sppr = spp / spp[ref0]

        self._beta['tot'] = sppr / (1 / beta_ref - (cumtrapz(x=self.rangebin, y=2 * self._lr['aer'] * sppr, initial=0)
                                                    - trapz(x=self.rangebin[:ref0], y=2 * self._lr['aer'] * sppr[:ref0])))

        self._beta['aer'] = self._beta['tot'] - self._beta['mol']

        self._alpha['aer'] = self._beta['aer'] * self._lr['aer']

        self._alpha['tot'] = self._alpha['mol'] + self._alpha['aer']

        self.tau = trapz(self._alpha["aer"], self.rangebin)

        return self._alpha["aer"].copy(), self._beta["aer"].copy(), self._lr["aer"]

    def _mc_fit(self):
        self._mc_bool = False

        original_signal = self.signal.copy()
        signals = np.random.randn(self.mc_iter, len(self.signal)) * self.uncertainty + self.signal

        betas = []
        alphas = []
        taus = []
        for signal in signals:
            self.signal = signal.copy()
            alpha, beta, _ = self.fit()
            alphas.append(alpha)
            betas.append(beta)
            taus.append(trapz(alphas[-1][self.tau_lims[0]: self.tau_lims[1]],
                              self.rangebin[self.tau_lims[0]: self.tau_lims[1]]))

        self._alpha['aer'] = np.mean(alphas, axis=0)
        self._alpha['tot'] = self._alpha['mol'] + self._alpha['aer']
        self._alpha_std = np.std(alphas, ddof=1, axis=0)
        self._beta['aer'] = np.mean(betas, axis=0)
        self._beta_std = np.std(betas, ddof=1, axis=0)
        self._beta['tot'] = self._beta['aer'] + self._beta['mol']
        self.tau = np.mean(taus)
        self.tau_std = np.std(taus, ddof=1)

        self.signal = original_signal.copy()
        self._mc_bool = True
        return (self._alpha["aer"].copy(),
                self._alpha_std.copy(),
                self._beta["aer"].copy(),
                self._beta_std.copy(),
                self._lr["aer"],
                self.tau,
                self.tau_std)
