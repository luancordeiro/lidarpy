import xarray as xr
import numpy as np
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular
from scipy.integrate import cumtrapz, trapz


def diff_(y, x, window=None):
    return np.gradient(y) / np.gradient(x)


def diff_sliding_lsq_fit(y, x, window=5):
    diff_y = []
    for i in range(window, len(y) + 1):
        x_fit = x[i - window:i]
        y_fit = y[i - window:i]

        a_ = np.vstack([x_fit, np.ones(len(x_fit))]).T
        a, _ = np.linalg.lstsq(a_, y_fit, rcond=None)[0]

        if i != window:
            diff_y.append(a)
        else:
            diff_y += [a] * window

    return np.array(diff_y)


class Raman:
    """
    Input:
    z                - Altitude                                          [m]
    elastic_signal   -
    inelastic_signal -
    lidar_wavelength - the wavelength of the lidar                       [m]
    raman_wavelength - the wavelength of the inelastic backscatter foton [m]
    angstrom_coeff   - dependence of                                     [u.a]
    p_air            - pressure                                          [Pa]
    t_air            - temperature                                       [K]
    co2ppmv          - co2 concentration                                 [ppmv]
    diff_strategy    - differentiation strategy for the calculation of aerosol extinction

    """
    _alpha = dict()
    _beta = dict()
    _diff_window = 7
    _diff_strategy = diff_sliding_lsq_fit

    def __init__(self,
                 lidar_data: xr.Dataset,
                 lidar_wavelength: float,
                 raman_wavelength: float,
                 angstrom_coeff: float,
                 p_air: np.array,
                 t_air: np.array,
                 ref: int = 4000,
                 pc: bool = True,
                 co2ppmv: int = 392):

        self.p_air, self.t_air = p_air, t_air
        self.lidar_wavelength, self.raman_wavelength = lidar_wavelength, raman_wavelength
        self.angstrom_coeff, self.co2ppmv = angstrom_coeff, co2ppmv

        self.z_ref = ref
        self.z = lidar_data.coords["Altitude"]

        self._ref = np.where(abs(self.z - self.z_ref) == min(abs(self.z - self.z_ref)))[0][0]
        self._delta_ref = np.where(abs(self.z - self.z_ref - 1500) == min(abs(self.z - self.z_ref - 1500)))[0][0]

        data_label = [f"{wave}_{int(pc)}" for wave in [lidar_wavelength, raman_wavelength]]
        self.elastic_signal = lidar_data.sel(wavelength=data_label[0]).data
        self.inelastic_signal = lidar_data.sel(wavelength=data_label[1]).data

    def get_alpha(self):
        return self._alpha.copy()

    def get_beta(self):
        return self._beta.copy()

    def get_lidar_ratio(self):
        return self._alpha["elastic_aer"] / self._beta["elastic_aer"]

    def _alpha_beta_molecular(self, co2ppmv):
        elastic_alpha_beta = AlphaBetaMolecular(self.p_air,
                                                self.t_air,
                                                self.lidar_wavelength,
                                                co2ppmv)

        inelastic_alpha_beta = AlphaBetaMolecular(self.p_air,
                                                  self.t_air,
                                                  self.raman_wavelength,
                                                  co2ppmv)

        self._alpha['elastic_mol'], self._beta['elastic_mol'], _ = elastic_alpha_beta.get_params()
        self._alpha['inelastic_mol'], self._beta['inelastic_mol'], _ = inelastic_alpha_beta.get_params()

    def _raman_scatterer_numerical_density(self) -> np.array:
        """Com base na eq. dos gases ideais e na razao de nitrogenio na atmosfera, calcula o perfil da densidade
        numerica utilizando os perfis de temperatura e pressao"""
        atm_numerical_density = self.p_air / (1.380649e-23 * self.t_air)
        return atm_numerical_density * 78.08e-2

    def _diff(self, y, x) -> np.array:
        """Realiza a suavizacao da curva e, em seguida, calcula a derivada necessaria para o calculo do coeficiente de
        extincao dos aerossois com base na estrategia escolhida."""
        return self._diff_strategy(y, x, self._diff_window)

    def _alpha_elastic_aer(self) -> np.array:
        """Retorna o coeficiente de extincao de aerossois."""
        ranged_corrected_signal = self.inelastic_signal * self.z ** 2

        y = np.log(self._raman_scatterer_numerical_density() / ranged_corrected_signal)

        diff_num_signal = self._diff(y, self.z)

        return (diff_num_signal - self._alpha['elastic_mol'] - self._alpha['inelastic_mol']) / \
               (1 + (self.lidar_wavelength / self.raman_wavelength) ** self.angstrom_coeff)

    def _alpha_elastic_total(self) -> np.array:
        return self._alpha["elastic_aer"] + self._alpha["elastic_mol"]

    def _alpha_inelastic_total(self) -> np.array:
        return self._alpha["inelastic_aer"] + self._alpha["inelastic_mol"]

    def _ref_value(self, y):
        p = np.poly1d(np.polyfit(self.z[self._ref - self._delta_ref: self._ref + self._delta_ref + 1],
                                 y[self._ref - self._delta_ref: self._ref + self._delta_ref + 1], 1))

        return p(self.z[self._ref])

    def _beta_elastic_total(self) -> np.array:
        scatterer_numerical_density = self._raman_scatterer_numerical_density()

        signal_ratio = ((self._ref_value(self.inelastic_signal) * self.elastic_signal /
                        (self._ref_value(self.elastic_signal) * self.inelastic_signal)) *
                        (scatterer_numerical_density / self._ref_value(scatterer_numerical_density)))

        attenuation_ratio = (np.exp(-cumtrapz(x=self.z, y=self._alpha_inelastic_total(), initial=0) +
                                    trapz(x=self.z[:self._ref], y=self._alpha_inelastic_total()[:self._ref])) /
                             np.exp(-cumtrapz(x=self.z, y=self._alpha_elastic_total(), initial=0) +
                                    trapz(x=self.z[:self._ref], y=self._alpha_elastic_total()[:self._ref])))

        beta_ref = self._ref_value(self._beta["elastic_mol"])

        return beta_ref * signal_ratio * attenuation_ratio

    def fit(self, diff_strategy=diff_sliding_lsq_fit, diff_window=7):
        self._diff_window = diff_window
        self._diff_strategy = diff_strategy

        self._alpha["elastic_aer"] = self._alpha_elastic_aer()

        self._beta["elastic_aer"] = self._beta_elastic_total() - self._beta["elastic_mol"]

        return self
