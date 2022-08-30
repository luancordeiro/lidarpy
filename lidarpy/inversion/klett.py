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
    print(f'mean signal: {mean_signal}')

    mean_model = np.mean(model[reference])
    model = model / mean_model
    print(f'mean_model: {mean_model}')

    coef_slope_linear, pcov = curve_fit(f=lambda x, a, b: a * x + b, xdata=model[reference], ydata=signal[reference])
    print(f'ab: {coef_slope_linear}')

    signal = (signal - coef_slope_linear[1]) / coef_slope_linear[0]

    coef_slope_linear = [coef_slope_linear[0] * mean_signal / mean_model, coef_slope_linear[1] * mean_signal]

    return signal, model, coef_slope_linear


def calib_strategy2(signal: np.array,
                    model: np.array,
                    reference: np.array):
    mean_signal = np.mean(signal[reference])
    signal = signal / mean_signal
    print(f'mean signal: {mean_signal}')

    mean_model = np.mean(model[reference])
    model = model / mean_model
    print(f'mean_model: {mean_model}')

    coef_slope, pcov = curve_fit(f=lambda x, a: a * x, xdata=model[reference], ydata=signal[reference])
    print(f'a: {coef_slope}')

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
    """
    _alpha = dict()
    _beta = dict()
    _lr = dict()
    fit_parameters = None
    _calib_strategies = {True: calib_strategy1, False: calib_strategy2}

    def __init__(self,
                 lidar_data: xr.Dataset,
                 ref: np.ndarray,
                 wavelength: int,
                 lidar_ratio: float,
                 p_air: np.ndarray,
                 t_air: np.ndarray,
                 pc: bool = True,
                 co2ppmv: int = 392,
                 correct_noise: bool = True):
        self.z = lidar_data.coords["altitude"]
        self.signal = lidar_data.sel(wavelength=f"{wavelength}_{int(pc)}").data
        self.ref = ref
        self._calib_strategy = self._calib_strategies[correct_noise]
        self._lr['aer'] = lidar_ratio

        self._get_molecular_alpha_beta(p_air, t_air, wavelength * 1e-9, co2ppmv)

    def __str__(self):
        return f"Lidar ratio = {self._lr['aer']}"

    def get_beta(self) -> dict:
        return self._beta.copy()

    def get_alpha(self) -> dict:
        return self._alpha.copy()

    def set_lidar_ratio(self, lidar_ratio):
        self._lr['aer'] = lidar_ratio
        return self

    def get_lidar_ratio(self) -> dict:
        return self._lr.copy()

    def get_model_mol(self) -> np.array:
        return self._beta['mol'] * np.exp(-2 * cumtrapz(self.z, self._alpha['mol'], initial=0)) / self.z ** 2

    def _get_molecular_alpha_beta(self, p_air, t_air, wavelength, co2ppmv):
        alpha_beta_mol = AlphaBetaMolecular(p_air, t_air, wavelength, co2ppmv)
        self._alpha['mol'], self._beta['mol'], self._lr['mol'] = alpha_beta_mol.get_params()

    def _calib(self):
        ref = np.round(self.ref / (self.z[1] - self.z[0]))
        ref = ref.astype(int)
        print(f'reference: {ref}')

        if len(ref) > 1:
            signal, model, self.fit_parameters = self._calib_strategy(signal=self.signal.copy(),
                                                                      model=self.get_model_mol(),
                                                                      reference=np.arange(ref[0], ref[-1], dtype=int))

            print(f'signal[R0]: {signal[ref[0]]}')
            print(f'model[R0]: {model[ref[0]]}')

            beta_ref = self._beta['mol'][ref[0]] * signal[ref[0]] / model[ref[0]]
            print(f'beta_ref: {beta_ref}')
        else:
            signal = self.signal
            beta_ref = self._beta['mol'][ref[0]]

        return beta_ref, signal, ref[0]

    def fit(self):
        print(f"lidar ratio: {self._lr['aer']}")
        beta_ref, signal, ref0 = self._calib()
        print()

        corrected_signal = signal * self.z ** 2
        print(f'corrected_singal: {corrected_signal[ref0 - 3:ref0 + 4]}')
        print()

        spp = corrected_signal * np.exp(- 2 * cumtrapz(x=self.z,
                                                       y=(self._lr['aer'] - self._lr['mol']) * self._beta['mol'],
                                                       initial=0))
        print(f'spp: {spp[ref0 - 3:ref0 + 4]}')
        print()
        sppr = spp / spp[ref0]

        print(f'sppr: {sppr[ref0 - 3: ref0 + 4]}')
        print()

        self._beta['tot'] = sppr / (1 / beta_ref - (cumtrapz(x=self.z, y=2 * self._lr['aer'] * sppr, initial=0)
                                                    - trapz(x=self.z[:ref0], y=2 * self._lr['aer'] * sppr[:ref0])))

        print(f"cumtrapz: {cumtrapz(x=self.z, y=2 * self._lr['aer'] * sppr, initial=0)[ref0 - 3: ref0 + 4]}")
        print()
        print(f"trapz: {trapz(x=self.z[:ref0], y=2 * self._lr['aer'] * sppr[:ref0])}")
        print()
        print(
            f"cumtrapz - trapz: {np.round((cumtrapz(x=self.z, y=2 * self._lr['aer'] * sppr, initial=0) - trapz(x=self.z[:ref0], y=2 * self._lr['aer'] * sppr[:ref0]))[ref0 - 3: ref0 + 4], 3)}")

        self._beta['aer'] = self._beta['tot'] - self._beta['mol']

        self._alpha['aer'] = self._beta['aer'] * self._lr['aer']

        self._alpha['tot'] = self._alpha['mol'] + self._alpha['aer']

        return self._alpha.copy(), self._beta.copy(), self._lr.copy()
