import xarray as xr
import numpy as np
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular
from scipy.integrate import cumtrapz, trapz
from sklearn.linear_model import LinearRegression


def diff_(y, x, window=None):
    return np.gradient(y) / np.gradient(x)


def diff_linear_regression(y: np.array, x: np.array, window: int = 5, weights: np.array = None):
    def fit(init, final):
        y_fit = y[init: final].reshape(-1, 1)
        x_fit = x[init: final].reshape(-1, 1)

        if weights is None:
            linear_regession = LinearRegression().fit(x_fit, y_fit)
        else:
            weight_fit = weights[init: final]
            linear_regession = LinearRegression().fit(x_fit, y_fit, sample_weight=weight_fit)

        return linear_regession.coef_[0][0]

    if window % 2 == 0:
        raise ValueError("window must be odd.")

    win = window // 2
    diff_y = []
    for i in range(win, len(y) - win - 10 - 1):
        diff_y.append(fit(i - win, i + win + 1))
#        if (i % 20 == 0) & (win <= window // 2 + 10):
#            win += 2

    for i in range(window // 2):
        # diff_y.insert(i, fit(None, i + window // 2))
        diff_y.insert(0, diff_y[0])

    while len(diff_y) != len(y):
        diff_y += [diff_y[-1]]

    return np.array(diff_y)


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

    def __init__(self, lidar_data: xr.DataArray, lidar_wavelength: int, raman_wavelength: int, angstrom_coeff: float,
                 p_air: np.array, t_air: np.array, z_ref: int, pc: bool = True, co2ppmv: int = 392, mc_iter: int = None,
                 tau_ind: np.array = None):
        if (mc_iter is not None) and (tau_ind is None):
            raise Exception("Para realizar mc, é necessário add mc_iter e tau_ind")

        data_label = [f"{wave}_{int(pc)}" for wave in [lidar_wavelength, raman_wavelength]]
        self.elastic_signal = lidar_data.sel(wavelength=data_label[0]).data
        self.inelastic_signal = lidar_data.sel(wavelength=data_label[1]).data
        self.z = lidar_data.coords["altitude"].data
        self.p_air = p_air
        self.t_air = t_air
        self.mc_iter = mc_iter
        self.tau_ind = tau_ind
        self.lidar_wavelength = lidar_wavelength * 1e-9
        self.raman_wavelength = raman_wavelength * 1e-9
        self.angstrom_coeff = angstrom_coeff
        self.co2ppmv = co2ppmv

        z_ref = lidar_data.coords["altitude"].sel(altitude=z_ref, method="nearest").data
        z_delta_ref = lidar_data.coords["altitude"].sel(altitude=z_ref - 2000, method="nearest").data
        self._ref = np.where(self.z == z_ref)[0][0]
        self._delta_ref = self._ref - np.where(self.z == z_delta_ref)[0][0]

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

    def _diff(self, num_density, ranged_corrected_signal, z) -> np.array:
        """Realiza a suavizacao da curva e, em seguida, calcula a derivada necessaria para o calculo do coeficiente de
        extincao dos aerossois com base na estrategia escolhida."""
        y = np.log(num_density / ranged_corrected_signal)
        return self._diff_strategy(y, z, self._diff_window)

    def _alpha_elastic_aer(self) -> np.array:
        """Retorna o coeficiente de extincao de aerossois."""
        diff_num_signal = self._diff(self._raman_scatterer_numerical_density(),
                                     self.inelastic_signal * self.z ** 2,
                                     self.z)

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

        signal_ratio = ((self._ref_value(self.inelastic_signal) * self.elastic_signal
                         / (self._ref_value(self.elastic_signal) * self.inelastic_signal))
                        * (scatterer_numerical_density / self._ref_value(scatterer_numerical_density)))

        attenuation_ratio = (np.exp(-cumtrapz(x=self.z, y=self._alpha_inelastic_total(), initial=0)
                                    + trapz(x=self.z[:self._ref], y=self._alpha_inelastic_total()[:self._ref]))
                             / np.exp(-cumtrapz(x=self.z, y=self._alpha_elastic_total(), initial=0)
                                      + trapz(x=self.z[:self._ref], y=self._alpha_elastic_total()[:self._ref])))

        beta_ref = self._ref_value(self._beta["elastic_mol"])

        return beta_ref * signal_ratio * attenuation_ratio

    def fit(self, diff_strategy=diff_linear_regression, diff_window=7):
        if (self.mc_iter is not None) & self._mc_bool:
            return self._mc_fit(diff_strategy=diff_linear_regression, diff_window=7)
        self._diff_window = diff_window
        self._diff_strategy = diff_strategy

        self._alpha["elastic_aer"] = self._alpha_elastic_aer()

        self._alpha["inelastic_aer"] = self._alpha["elastic_aer"] / (
                self.raman_wavelength / self.lidar_wavelength) ** self.angstrom_coeff

        self._beta["elastic_aer"] = self._beta_elastic_total() - self._beta["elastic_mol"]

        return (self._alpha["elastic_aer"],
                self._beta["elastic_aer"],
                self._alpha["elastic_aer"] / self._beta["elastic_aer"])

    def _mc_fit(self, diff_strategy, diff_window):
        self._mc_bool = False

        original_elastic_signal = self.elastic_signal.copy()
        original_inelastic_signal = self.inelastic_signal.copy()

        elastic_signals = np.random.poisson(self.elastic_signal, size=(self.mc_iter, len(self.elastic_signal)))
        inelastic_signals = np.random.poisson(self.inelastic_signal, size=(self.mc_iter, len(self.inelastic_signal)))

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
            taus.append(trapz(alphas[-1][self.tau_ind], self.z[self.tau_ind]))

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
