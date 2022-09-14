import xarray as xr
import numpy as np
from scipy.integrate import trapz
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular
import matplotlib.pyplot as plt


class Transmittance:
    _alpha = None
    _beta = None
    tau = None

    def __init__(self,
                 lidar_data: xr.DataArray,
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

        z_lims += [z_lims[0] - 500, z_lims[1] + 500]
        print(z_lims)

        z_ref = lidar_data.coords["altitude"].sel(altitude=z_lims, method="nearest").data

        self._ref_base = np.where((self.z == z_ref[0]) | (self.z == z_ref[2]))[0]
        self._ref_top = np.where((self.z == z_ref[1]) | (self.z == z_ref[3]))[0]

        print(self._ref_base)
        print(self._ref_top)
        print(self.z[self._ref_base])
        print(self.z[self._ref_top])

        self._get_molecular_alpha_beta(p_air, t_air, wavelength * 1e-9, co2ppmv)

    def _range_corrected_signal(self):
        return self.signal * self.z ** 2

    def _get_molecular_alpha_beta(self, p_air, t_air, wavelength, co2ppmv):
        alpha_beta_mol = AlphaBetaMolecular(p_air, t_air, wavelength, co2ppmv)
        self._alpha, self._beta, _ = alpha_beta_mol.get_params()

    def _ref_value(self, y, ref):
        p = np.poly1d(np.polyfit(self.z[ref[0]: ref[1]], y[ref[0]: ref[1]], 1))

        plt.plot(self.z[ref[0]: ref[1]], y[ref[0]: ref[1]], ".")
        plt.plot(self.z[ref[0]: ref[1]], p(self.z[ref[0]: ref[1]]), "-")
        plt.plot(self.z[(ref[0] + ref[1]) // 2], p(self.z[(ref[0] + ref[1]) // 2]), "*")
        plt.show()

        return p(self.z[(ref[0] + ref[1]) // 2])

    def fit(self):
        s = self._range_corrected_signal()
        s0, s1 = self._ref_value(s, self._ref_base), self._ref_value(s, self._ref_top)
        beta0, beta1 = self._ref_value(self._beta, self._ref_base), self._ref_value(self._beta, self._ref_top)

        transmittance_factor = (s1 * beta0 / (s0 * beta1)
                                * np.exp(2 * trapz(x=self.z[(self._ref_base[0] + self._ref_base[1]) // 2:
                                                            (self._ref_top[0] + self._ref_top[1]) // 2],
                                                   y=self._alpha[(self._ref_base[0] + self._ref_base[1]) // 2:
                                                                 (self._ref_top[0] + self._ref_top[1]) // 2])))

        self.tau = -0.5 * np.log(transmittance_factor)

        return self.tau
