import xarray as xr
import numpy as np
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular
from lidarpy.data.manipulation import z_finder, filter_wavelength
from lidarpy.data.raman_smoothers import diff_linear_regression
from scipy.integrate import cumtrapz, trapz
from scipy.signal import savgol_filter


class Raman:
    """
    Input:
    z                - Altitude                                          [m]
    elastic_signal   -
    inelastic_signal -
    lidar_wavelength - the wavelength of the lidar                       [nm]
    raman_wavelength - the wavelength of the inelastic backscatter foton [nm]
    angstrom_coeff   - dependence of                                     [u.a]
    p_air            - pressure                                          [Pa]
    t_air            - temperature                                       [K]
    co2ppmv          - co2 concentration                                 [ppmv]
    diff_strategy    - differentiation strategy for the calculation of aerosol extinction

    """
    _alpha = dict()
    _alpha_std = None
    _beta = dict()
    _beta_std = None
    _diff_window = 7
    _diff_strategy = diff_linear_regression
    _mc_bool = True

    def __init__(self, lidar_data: xr.Dataset, lidar_wavelength: int, raman_wavelength: int, angstrom_coeff: float,
                 p_air: np.array, t_air: np.array, z_ref: list, pc: bool = True, co2ppmv: int = 392,
                 mc_iter: int = None, tau_ind: np.array = None):
        if (mc_iter is not None) and (tau_ind is None):
            raise Exception("Para realizar mc, é necessário add mc_iter e tau_ind")
        self.elastic_signal = filter_wavelength(lidar_data, lidar_wavelength, pc)
        self.inelastic_signal = filter_wavelength(lidar_data, raman_wavelength, pc)
        self.elastic_uncertainty = (filter_wavelength(lidar_data, lidar_wavelength, pc, "sigma")
                                    if "sigma" in lidar_data.variables else None)
        self.inelastic_uncertainty = (filter_wavelength(lidar_data, raman_wavelength, pc, "sigma")
                                      if "sigma" in lidar_data.variables else None)
        self.rangebin = lidar_data.coords["rangebin"].data
        self.p_air = p_air
        self.t_air = t_air
        self.mc_iter = mc_iter
        self.tau_ind = tau_ind
        self.lidar_wavelength = lidar_wavelength * 1e-9
        self.raman_wavelength = raman_wavelength * 1e-9
        self.angstrom_coeff = angstrom_coeff
        self.co2ppmv = co2ppmv
        self._ref = z_finder(self.rangebin, z_ref)
        self._mean_ref = (self._ref[0] + self._ref[1]) // 2
        self._get_alpha_beta_molecular(co2ppmv)

    def get_alpha(self):
        return self._alpha.copy()

    def get_beta(self):
        return self._beta.copy()

    def get_lidar_ratio(self):
        return self._alpha["elastic_aer"] / self._beta["elastic_aer"]

    def _get_alpha_beta_molecular(self, co2ppmv):
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

    def _diff(self, num_density, ranged_corrected_signal) -> np.array:
        """Realiza a suavizacao da curva e, em seguida, calcula a derivada necessaria para o calculo do coeficiente de
        extincao dos aerossois com base na estrategia escolhida."""
        return self._diff_strategy(num_density,
                                   ranged_corrected_signal,
                                   self.rangebin,
                                   self._diff_window,
                                   self.inelastic_uncertainty)

    def _alpha_elastic_aer(self) -> np.array:
        """Retorna o coeficiente de extincao de aerossois."""
        diff_num_signal = self._diff(self._raman_scatterer_numerical_density(),
                                     self.inelastic_signal * self.rangebin ** 2)

        return (diff_num_signal - self._alpha['elastic_mol'] - self._alpha['inelastic_mol']) / \
               (1 + (self.lidar_wavelength / self.raman_wavelength) ** self.angstrom_coeff)

    def _alpha_elastic_total(self) -> np.array:
        return self._alpha["elastic_aer"] + self._alpha["elastic_mol"]

    def _alpha_inelastic_total(self) -> np.array:
        return self._alpha["inelastic_aer"] + self._alpha["inelastic_mol"]

    def _ref_value(self, y):
        p = np.poly1d(np.polyfit(self.rangebin[self._ref[0]: self._ref[1] + 1],
                                 y[self._ref[0]: self._ref[1] + 1], 1))

        return p(self.rangebin[self._mean_ref])

    def _beta_elastic_total(self) -> np.array:
        scatterer_numerical_density = self._raman_scatterer_numerical_density()

        signal_ratio = ((self._ref_value(self.inelastic_signal) * self.elastic_signal
                         / (self._ref_value(self.elastic_signal) * self.inelastic_signal))
                        * (scatterer_numerical_density / self._ref_value(scatterer_numerical_density)))

        attenuation_ratio = (np.exp(-cumtrapz(x=self.rangebin, y=self._alpha_inelastic_total(), initial=0)
                                    + trapz(x=self.rangebin[:self._mean_ref + 1],
                                            y=self._alpha_inelastic_total()[:self._mean_ref + 1]))
                             / np.exp(-cumtrapz(x=self.rangebin, y=self._alpha_elastic_total(), initial=0)
                                      + trapz(x=self.rangebin[:self._mean_ref + 1],
                                              y=self._alpha_elastic_total()[:self._mean_ref + 1])))

        beta_ref = self._ref_value(self._beta["elastic_mol"])

        return beta_ref * signal_ratio * attenuation_ratio

    def fit(self, diff_strategy=diff_linear_regression, diff_window=7, beta_smoother=lambda x: x):
        if (self.mc_iter is not None) & self._mc_bool:
            return self._mc_fit(diff_strategy=diff_strategy, diff_window=diff_window)
        self._diff_window = diff_window
        self._diff_strategy = diff_strategy

        self._alpha["elastic_aer"] = self._alpha_elastic_aer()

        self._alpha["inelastic_aer"] = self._alpha["elastic_aer"] / (
                self.raman_wavelength / self.lidar_wavelength) ** self.angstrom_coeff

        self._beta["elastic_aer"] = self._beta_elastic_total() - self._beta["elastic_mol"]

        n_sg = round(0.625 * self._diff_window + 0.23)

        return (
            self._alpha["elastic_aer"],
            self._beta["elastic_aer"],
            self._alpha["elastic_aer"] / beta_smoother(savgol_filter(self._beta["elastic_aer"], n_sg, 2))
        )

    def _mc_fit(self, diff_strategy, diff_window):
        self._mc_bool = False

        original_elastic_signal = self.elastic_signal.copy()
        original_inelastic_signal = self.inelastic_signal.copy()

        elastic_signals = (np.random.randn(self.mc_iter, len(self.elastic_signal)) * self.elastic_uncertainty
                           + self.elastic_signal)
        inelastic_signals = (np.random.randn(self.mc_iter, len(self.inelastic_signal)) * self.inelastic_uncertainty
                             + self.inelastic_signal)

        alphas = []
        betas = []
        lrs = []
        taus = []
        for elastic_signal, inelastic_signal in zip(elastic_signals, inelastic_signals):
            self.elastic_signal = elastic_signal.copy()
            self.inelastic_signal = inelastic_signal.copy()

            alpha, beta, lr = self.fit(diff_strategy, diff_window)

            alphas.append(alpha)
            betas.append(beta)
            lrs.append(lr)
            taus.append(trapz(alphas[-1][self.tau_ind], self.rangebin[self.tau_ind]))

        self._alpha["elastic_aer"] = np.mean(alphas, axis=0)
        self._alpha_std = np.std(alphas, ddof=1, axis=0)
        self._beta["elastic_aer"] = np.mean(betas, axis=0)
        self._beta_std = np.std(betas, ddof=1, axis=0)
        self._lidar_ratio = np.mean(lrs, axis=0)
        self._lidar_ratio_std = np.std(lrs, ddof=1, axis=0)
        self.tau = np.mean(taus, axis=0)
        self.tau_std = np.std(taus, ddof=1, axis=0)

        self.elastic_signal = original_elastic_signal.copy()
        self.inelastic_signal = original_inelastic_signal.copy()
        self._mc_bool = True

        return (self._alpha["elastic_aer"].copy(),
                self._alpha_std.copy(),
                self._beta["aer"].copy(),
                self._beta_std.copy(),
                self._lidar_ratio,
                self._lidar_ratio_std,
                self.tau,
                self.tau_std)


class Raman2:
    """
    Input:
    z                - Altitude                                          [m]
    elastic_signal   -
    inelastic_signal -
    lidar_wavelength - the wavelength of the lidar                       [nm]
    raman_wavelength - the wavelength of the inelastic backscatter foton [nm]
    angstrom_coeff   - dependence of                                     [u.a]
    p_air            - pressure                                          [Pa]
    t_air            - temperature                                       [K]
    co2ppmv          - co2 concentration                                 [ppmv]
    diff_strategy    - differentiation strategy for the calculation of aerosol extinction

    """
    _alpha = dict()
    _alpha_std = None
    _beta = dict()
    _beta_std = None
    _diff_window = 7
    _diff_strategy = diff_linear_regression
    _mc_bool = True
    _lr = None

    def __init__(self, lidar_data: xr.Dataset, lidar_wavelength: int, raman_wavelength: int, angstrom_coeff: float,
                 p_air: np.array, t_air: np.array, z_ref: list, pc: bool = True, co2ppmv: int = 392,
                 mc_iter: int = None, tau_ind: np.array = None):
        if (mc_iter is not None) and (tau_ind is None):
            raise Exception("Para realizar mc, é necessário add mc_iter e tau_ind")
        self.elastic_signal = filter_wavelength(lidar_data, lidar_wavelength, pc)
        self.inelastic_signal = filter_wavelength(lidar_data, raman_wavelength, pc)
        self.elastic_uncertainty = (filter_wavelength(lidar_data, lidar_wavelength, pc, "sigma")
                                    if "sigma" in lidar_data.variables else None)
        self.inelastic_uncertainty = (filter_wavelength(lidar_data, raman_wavelength, pc, "sigma")
                                      if "sigma" in lidar_data.variables else None)
        self.rangebin = lidar_data.coords["rangebin"].data
        self.p_air = p_air
        self.t_air = t_air
        self.mc_iter = mc_iter
        self.tau_ind = tau_ind
        self.lidar_wavelength = lidar_wavelength * 1e-9
        self.raman_wavelength = raman_wavelength * 1e-9
        self.angstrom_coeff = angstrom_coeff
        self.co2ppmv = co2ppmv
        self._ref = z_finder(self.rangebin, z_ref)
        self._mean_ref = (self._ref[0] + self._ref[1]) // 2
        self._get_alpha_beta_molecular(co2ppmv)

    def get_alpha(self):
        return self._alpha.copy()

    def get_beta(self):
        return self._beta.copy()

    def get_lidar_ratio(self):
        return self._lr.copy()

    def _get_alpha_beta_molecular(self, co2ppmv):
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

    def _diff(self, num_density, ranged_corrected_signal) -> np.array:
        """Realiza a suavizacao da curva e, em seguida, calcula a derivada necessaria para o calculo do coeficiente de
        extincao dos aerossois com base na estrategia escolhida."""
        return self._diff_strategy(num_density,
                                   ranged_corrected_signal,
                                   self.rangebin,
                                   self._diff_window,
                                   self.inelastic_uncertainty)

    def _alpha_elastic_aer(self) -> np.array:
        """Retorna o coeficiente de extincao de aerossois."""
        diff_num_signal = self._diff(self._raman_scatterer_numerical_density(),
                                     self.inelastic_signal * self.rangebin ** 2)

        return (diff_num_signal - self._alpha['elastic_mol'] - self._alpha['inelastic_mol']) / \
               (1 + (self.lidar_wavelength / self.raman_wavelength) ** self.angstrom_coeff)

    def _alpha_elastic_total(self) -> np.array:
        return self._alpha["elastic_aer"] + self._alpha["elastic_mol"]

    def _alpha_inelastic_total(self) -> np.array:
        return self._alpha["inelastic_aer"] + self._alpha["inelastic_mol"]

    def _ref_value(self, y):
        p = np.poly1d(np.polyfit(self.rangebin[self._ref[0]: self._ref[1] + 1],
                                 y[self._ref[0]: self._ref[1] + 1], 1))

        return p(self.rangebin[self._mean_ref])

    def _beta_elastic_total(self) -> np.array:
        scatterer_numerical_density = self._raman_scatterer_numerical_density()

        signal_ratio = ((self._ref_value(self.inelastic_signal) * self.elastic_signal
                         / (self._ref_value(self.elastic_signal) * self.inelastic_signal))
                        * (scatterer_numerical_density / self._ref_value(scatterer_numerical_density)))

        attenuation_ratio = (np.exp(-cumtrapz(x=self.rangebin, y=self._alpha_inelastic_total(), initial=0)
                                    + trapz(x=self.rangebin[:self._mean_ref + 1],
                                            y=self._alpha_inelastic_total()[:self._mean_ref + 1]))
                             / np.exp(-cumtrapz(x=self.rangebin, y=self._alpha_elastic_total(), initial=0)
                                      + trapz(x=self.rangebin[:self._mean_ref + 1],
                                              y=self._alpha_elastic_total()[:self._mean_ref + 1])))

        beta_ref = self._ref_value(self._beta["elastic_mol"])

        return beta_ref * signal_ratio * attenuation_ratio

    def fit(self, diff_strategy=diff_linear_regression, diff_window=7, beta_smoother=lambda x: x,
            extinction_smoother=lambda x: x):
        if (self.mc_iter is not None) & self._mc_bool:
            return self._mc_fit(diff_strategy=diff_strategy, diff_window=diff_window)
        self._diff_window = diff_window
        self._diff_strategy = diff_strategy

        self._alpha["elastic_aer"] = extinction_smoother(self._alpha_elastic_aer())

        self._alpha["inelastic_aer"] = self._alpha["elastic_aer"] / (
                self.raman_wavelength / self.lidar_wavelength) ** self.angstrom_coeff

        self._beta["elastic_aer"] = extinction_smoother(self._beta_elastic_total() - self._beta["elastic_mol"])

        n_sg = round(0.625 * self._diff_window + 0.23)

        self._lr = self._alpha["elastic_aer"] / beta_smoother(savgol_filter(self._beta["elastic_aer"], n_sg, 2))

        return (
            self._alpha["elastic_aer"],
            self._beta["elastic_aer"],
            self._alpha["elastic_aer"] / beta_smoother(savgol_filter(self._beta["elastic_aer"], n_sg, 2))
        )

    def _mc_fit(self, diff_strategy, diff_window):
        self._mc_bool = False

        original_elastic_signal = self.elastic_signal.copy()
        original_inelastic_signal = self.inelastic_signal.copy()

        elastic_signals = (np.random.randn(self.mc_iter, len(self.elastic_signal)) * self.elastic_uncertainty
                           + self.elastic_signal)
        inelastic_signals = (np.random.randn(self.mc_iter, len(self.inelastic_signal)) * self.inelastic_uncertainty
                             + self.inelastic_signal)

        alphas = []
        betas = []
        lrs = []
        taus = []
        for elastic_signal, inelastic_signal in zip(elastic_signals, inelastic_signals):
            self.elastic_signal = elastic_signal.copy()
            self.inelastic_signal = inelastic_signal.copy()

            alpha, beta, lr = self.fit(diff_strategy, diff_window)

            alphas.append(alpha)
            betas.append(beta)
            lrs.append(lr)
            taus.append(trapz(alphas[-1][self.tau_ind], self.rangebin[self.tau_ind]))

        self._alpha["elastic_aer"] = np.mean(alphas, axis=0)
        self._alpha_std = np.std(alphas, ddof=1, axis=0)
        self._beta["elastic_aer"] = np.mean(betas, axis=0)
        self._beta_std = np.std(betas, ddof=1, axis=0)
        self._lidar_ratio = np.mean(lrs, axis=0)
        self._lidar_ratio_std = np.std(lrs, ddof=1, axis=0)
        self.tau = np.mean(taus, axis=0)
        self.tau_std = np.std(taus, ddof=1, axis=0)

        self.elastic_signal = original_elastic_signal.copy()
        self.inelastic_signal = original_inelastic_signal.copy()
        self._mc_bool = True

        return (self._alpha["elastic_aer"].copy(),
                self._alpha_std.copy(),
                self._beta["aer"].copy(),
                self._beta_std.copy(),
                self._lidar_ratio,
                self._lidar_ratio_std,
                self.tau,
                self.tau_std)
