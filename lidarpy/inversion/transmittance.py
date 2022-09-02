import xarray as xr
import numpy as np
from scipy.integrate import trapz
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular


class Transmittance:
    _alpha = None
    _beta = None
    tau = None

    def __init__(self,
                 lidar_data: xr.Dataset,
                 z_lims: list,
                 wavelength: int,
                 p_air: np.ndarray,
                 t_air: np.ndarray,
                 pc=True,
                 co2ppmv: int = 392,):
        if "wavelength" in lidar_data.dims:
            self.signal = lidar_data.sel(wavelength=f"{wavelength}_{int(pc)}").data
        else:
            self.signal = lidar_data.data

        self.z = lidar_data.coords["altitude"].data

        z_ref = lidar_data.coords["altitude"].sel(altitude=z_lims, method="nearest").data
        self._ref = np.where((self.z == z_ref[0]) | (self.z == z_ref[1]))[0]

        self._get_molecular_alpha_beta(p_air, t_air, wavelength * 1e-9, co2ppmv)

    def _range_corrected_signal(self):
        return self.signal * self.z ** 2

    def _get_molecular_alpha_beta(self, p_air, t_air, wavelength, co2ppmv):
        alpha_beta_mol = AlphaBetaMolecular(p_air, t_air, wavelength, co2ppmv)
        self._alpha, self._beta, _ = alpha_beta_mol.get_params()

    def fit(self, delta_z=10):
        """Para calcular o tau, estou pegando o sinal médio dos 10 bins abaixo da camada
        e os 10 bins diretamente acima da camada"""
        s = self._range_corrected_signal()
        s0, s1 = s[self._ref[0] - delta_z:self._ref[0]].mean(), s[self._ref[1]:self._ref[1] + delta_z].mean()
        beta0, beta1 = (self._beta[self._ref[0] - 10:self._ref[0]].mean(),
                        self._beta[self._ref[1]:self._ref[1] + 10].mean())

        transmittance_factor = (s1 * beta0 / (s0 * beta1)
                                * np.exp(2 * trapz(x=self.z[self._ref[0]:self._ref[1] + 1],
                                                   y=self._alpha[self._ref[0]:self._ref[1] + 1])))

        self.tau = -0.5 * np.log(transmittance_factor)

        return self.tau
