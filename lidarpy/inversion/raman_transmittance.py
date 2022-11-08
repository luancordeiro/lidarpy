import xarray as xr
import numpy as np
from lidarpy.data.manipulation import molecular_raman_model, z_finder, filter_wavelength
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular
import matplotlib.pyplot as plt


class GetCod:
    cod = None
    cod_std = None
    tau = {}
    _mc_bool = True

    def __init__(self, lidar_data: xr.Dataset, cloud_lims: list, lidar_wavelength: int, raman_wavelength: int,
                 angstrom_coeff: float, p_air: np.ndarray, t_air: np.ndarray, pc=True, co2ppmv: int = 392,
                 fit_delta_z=2000, mc_iter=None):
        self.rangebin = lidar_data.coords["rangebin"].data
        self.lidar_data = lidar_data
        self.cloud_lims = cloud_lims
        self.lidar_wavelength = lidar_wavelength
        self.raman_wavelength = raman_wavelength
        self.angstrom_coeff = angstrom_coeff
        self.p_air = p_air
        self.t_air = t_air
        self.pc = pc
        self.co2ppmv = co2ppmv
        self.fit_delta_z = fit_delta_z
        self.mc_iter = mc_iter

        self.cloud_ref = z_finder(self.rangebin, cloud_lims)
        alpha = {}
        alpha['elastic_mol'], *_ = AlphaBetaMolecular(p_air,
                                                      t_air,
                                                      lidar_wavelength,
                                                      co2ppmv).get_params()

        alpha['inelastic_mol'], *_ = AlphaBetaMolecular(p_air,
                                                        t_air,
                                                        raman_wavelength,
                                                        co2ppmv).get_params()

        for key, value in alpha.items():
            self.tau[key] = np.trapz(value[self.cloud_ref[0]:self.cloud_ref[1]])

        self.scatterer_numerical_density = 78.08e-2 * p_air / (1.380649e-23 * t_air)

    def fit(self):
        if (self.mc_iter is not None) & self._mc_bool:
            return self._mc_fit()
        fit_base_ref = [self.cloud_lims[0] - self.fit_delta_z - 50, self.cloud_lims[0] - 50]

        fit_top_ref = [self.cloud_lims[1] + 50, self.cloud_lims[1] + self.fit_delta_z + 50]

        model_rcs_base = molecular_raman_model(self.lidar_data, 355, 387, self.p_air, self.t_air, fit_base_ref,
                                               self.co2ppmv, self.pc) * self.rangebin ** 2

        model_rcs_top = molecular_raman_model(self.lidar_data, 355, 387, self.p_air, self.t_air, fit_top_ref,
                                              self.co2ppmv, self.pc) * self.rangebin ** 2

        # fit_b_ref = z_finder(self.rangebin, fit_base_ref)
        # fit_t_ref = z_finder(self.rangebin, fit_top_ref)
        # p_ref = z_finder(self.rangebin, [fit_base_ref[0] - 3000, fit_top_ref[1] + 3000])
        # rcs = filter_wavelength(self.lidar_data, self.raman_wavelength, self.pc) * self.rangebin ** 2
        # plt.plot(self.rangebin, rcs, "b-", label="RCS 387")
        # plt.plot(self.rangebin[fit_b_ref[0]:fit_b_ref[1]], rcs[fit_b_ref[0]:fit_b_ref[1]], "y*",
        #          label="base reference fit")
        # plt.plot(self.rangebin[fit_t_ref[0]:fit_t_ref[1]], rcs[fit_t_ref[0]:fit_t_ref[1]], "g*",
        #          label="top reference fit")
        # plt.plot(self.rangebin[p_ref[0]:p_ref[1]], model_rcs_base[p_ref[0]:p_ref[1]], "k--", label="base fit")
        # plt.plot(self.rangebin[p_ref[0]:p_ref[1]], model_rcs_top[p_ref[0]:p_ref[1]], "r--", label="top fit")
        # plt.legend()
        # plt.grid()
        # plt.show()

        tau_cloud = (np.log(self.scatterer_numerical_density[self.cloud_ref[1]] * model_rcs_base[self.cloud_ref[0]]
                            / (self.scatterer_numerical_density[self.cloud_ref[0]] * model_rcs_top[self.cloud_ref[1]]))
                     - self.tau["elastic_mol"] - self.tau["inelastic_mol"])

        self.cod = tau_cloud / (1 + (self.lidar_wavelength / self.raman_wavelength) ** self.angstrom_coeff)

        return self.cod

    def _mc_fit(self):
        self._mc_bool = False

        original_ds = self.lidar_data.copy()
        original_inelastic_signal = filter_wavelength(self.lidar_data, self.lidar_wavelength, self.pc)
        inelastic_uncertainty = filter_wavelength(self.lidar_data, self.raman_wavelength, self.pc, "sigma")

        inelastic_signals = (np.random.randn(self.mc_iter, len(original_inelastic_signal)) * inelastic_uncertainty
                             + original_inelastic_signal)

        self.lidar_data = self.lidar_data.sel(channel=f"{self.raman_wavelength}_{int(self.pc)}")

        taus = []
        for inelastic_signal in inelastic_signals:
            self.lidar_data.phy.data = inelastic_signal
            taus.append(self.fit())

        self.cod = np.mean(taus, axis=0)
        self.cod_std = np.std(taus, ddof=1, axis=0)

        self.lidar_data = original_ds
        self._mc_bool = True

        return self.cod, self.cod_std
