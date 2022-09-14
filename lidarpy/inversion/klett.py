import numpy as np
import xarray as xr
from scipy.integrate import cumtrapz, trapz
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular
from lidarpy.inversion.transmittance import Transmittance
import matplotlib.pyplot as plt


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

    def __init__(self, lidar_data: xr.DataArray, wavelength: int, p_air: np.ndarray, t_air: np.ndarray, z_ref: list,
                 lidar_ratio: float = None, pc: bool = True, co2ppmv: int = 392, correct_noise: bool = True,
                 mc_iter: int = None, tau_ind: np.array = None, z_lims: list = None):
        if wavelength in lidar_data.dims:
            self.signal = lidar_data.sel(wavelength=f"{wavelength}_{int(pc)}").data
        else:
            self.signal = lidar_data.data

        if (mc_iter is not None) & (tau_ind is None):
            raise Exception("Para realizar mc, é necessário add mc_iter e tau_ind")

        if (lidar_ratio is None) & ((z_lims is None) | (tau_ind is None)):
            raise Exception("Para realizar o calculo da razão lidar utilizando o método da transmitância é preciso "
                            "definir o z_lims e tau_ind")

        self.z = lidar_data.coords["altitude"].data
        self.tau_ind = tau_ind
        self.ref = lidar_data.coords["altitude"].sel(altitude=z_ref, method="nearest").data
        self._calib_strategy = self._calib_strategies[correct_noise]

        self._get_alpha_beta_molecular(p_air, t_air, wavelength * 1e-9, co2ppmv)

        self._lr['aer'] = (self._transmittance_lr(lidar_data, tau_ind, z_lims, wavelength, p_air, t_air, pc, co2ppmv)
                           if lidar_ratio is None else lidar_ratio)
        self.mc_iter = mc_iter

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
                                                                      reference=np.arange(*ref))
            beta_ref = self._beta['mol'][ref[0]] * signal[ref[0]] / model[ref[0]]
        else:
            signal = signal
            beta_ref = self._beta['mol'][ref[0]]

        return beta_ref, signal, ref[0]

    def _transmittance_lr(self, da, tau_ind, z_lims, wavelength, p_air, t_air, pc, co2ppmv) -> float:
        tau_transmittance = Transmittance(da,
                                          z_lims,
                                          wavelength,
                                          p_air,
                                          t_air,
                                          pc,
                                          co2ppmv).fit()

        print(f"tau transmittance = {tau_transmittance.round(2)}")

        taus = []
        lidar_ratios = np.arange(5, 75, 5)
        for lr in lidar_ratios:
            self._lr['aer'] = lr

            alpha, *_ = self.fit()

            taus.append(trapz(alpha[tau_ind], self.z[tau_ind]))

        print(f"taus = {np.round(taus, 2)}")

        difference = (np.array(taus) - tau_transmittance) ** 2

        print(f"diff = {np.round(difference, 2)}")

        f_diff = interp1d(lidar_ratios, difference, kind="quadratic")

        new_lr = np.linspace(5, 70, 100)

        new_diff = f_diff(new_lr)

        plt.plot(new_lr, new_diff, "o")
        plt.plot(new_lr[new_diff.argmin()], min(new_diff), "*")
        plt.title(f"lidar ratio = {new_lr[new_diff.argmin()].round(2)}")
        plt.grid()
        plt.xlabel("Lidar ratio")
        plt.ylabel("Difference")
        plt.show()

        return new_lr[new_diff.argmin()]

    def fit(self):
        if (self.mc_iter is not None) & self._mc_bool:
            return self._mc_fit()

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

        self.tau = trapz(self._alpha["aer"], self.z)

        return self._alpha["aer"].copy(), self._beta["aer"].copy(), self._lr["aer"]

    def _mc_fit(self):
        self._mc_bool = False

        original_signal = self.signal.copy()
        signals = np.random.poisson(self.signal, size=(self.mc_iter, len(self.signal)))

        betas = []
        alphas = []
        taus = []
        for signal in signals:
            self.signal = signal.copy()
            alpha, beta, _ = self.fit()
            alphas.append(alpha)
            betas.append(beta)
            taus.append(trapz(alphas[-1][self.tau_ind], self.z[self.tau_ind]))

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
