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

    def fit0(self):
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

    def fit1(self):
        p0, p1 = self._ref_value(self.signal, self._ref_base), self._ref_value(self.signal, self._ref_top)
        s0 = p0 * self.z[(self._ref_base[0] + self._ref_base[1]) // 2] ** 2
        s1 = p1 * self.z[(self._ref_top[0] + self._ref_top[1]) // 2] ** 2
        beta0, beta1 = self._ref_value(self._beta, self._ref_base), self._ref_value(self._beta, self._ref_top)

        transmittance_factor = (s1 * beta0 / (s0 * beta1)
                                * np.exp(2 * trapz(x=self.z[(self._ref_base[0] + self._ref_base[1]) // 2:
                                                            (self._ref_top[0] + self._ref_top[1]) // 2],
                                                   y=self._alpha[(self._ref_base[0] + self._ref_base[1]) // 2:
                                                                 (self._ref_top[0] + self._ref_top[1]) // 2])))

        self.tau = -0.5 * np.log(transmittance_factor)

        return self.tau

    def fit2(self):
        s = self._range_corrected_signal()
        ref_base = self._ref_base[1]
        ref_top = self._ref_top[0]
        s0, s1 = s[ref_base - 5: ref_base + 1].mean(), s[ref_top: ref_top + 5].mean()
        beta0, beta1 = self._beta[ref_base - 5: ref_base + 1].mean(), self._beta[ref_top: ref_top + 5].mean()

        transmittance_factor = (s1 * beta0 / (s0 * beta1)
                                * np.exp(2 * trapz(x=self.z[ref_base: ref_top],
                                                   y=self._alpha[ref_base: ref_top])))

        self.tau = -0.5 * np.log(transmittance_factor)

        return self.tau

    def fit(self):
        ref_base = self._ref_base[1]
        ref_top = self._ref_top[0]
        p0, p1 = self.signal[ref_base - 15: ref_base + 1].mean(), self.signal[ref_top: ref_top + 16].mean()
        s0, s1 = p0 * self.z[ref_base] ** 2, p1 * self.z[ref_top] ** 2
        beta0, beta1 = self._beta[ref_base - 5: ref_base + 1].mean(), self._beta[ref_top: ref_top + 5].mean()

        transmittance_factor = (s1 * beta0 / (s0 * beta1)
                                * np.exp(2 * trapz(x=self.z[ref_base: ref_top],
                                                   y=self._alpha[ref_base: ref_top])))

        self.tau = -0.5 * np.log(transmittance_factor)

        return self.tau

    def fit3(self):
        s = self._range_corrected_signal()
        ref_base = self._ref_base[1]
        ref_top = self._ref_top[0]

        s0 = s[ref_base - 5: ref_base + 1].mean()
        beta0 = self._beta[ref_base - 5: ref_base + 1].mean()

        s1_list = s[ref_top: ref_top + 50]
        beta1_list = self._beta[ref_top: ref_top + 50]

        transmittances = []
        taus = []
        for i, (s1, beta1) in enumerate(zip(s1_list, beta1_list)):
            transmittance_factor = (s1 * beta0 / (s0 * beta1)
                                    * np.exp(2 * trapz(x=self.z[ref_base:ref_top + i],
                                                       y=self._alpha[ref_base:ref_top + i])))
            transmittances.append(transmittance_factor)
            taus.append(-0.5 * np.log(transmittance_factor))

        plt.plot(s1_list, transmittances, ".", label=f"tau={np.mean(taus).round(2)} +- {np.std(taus, ddof=1).round(2)}")
        plt.grid()
        plt.legend()
        plt.show()

        plt.plot(s1_list, taus, ".", label=f"tau={np.mean(taus).round(2)} +- {np.std(taus, ddof=1).round(2)}")
        plt.grid()
        plt.legend()
        plt.show()

        return np.mean(taus)
