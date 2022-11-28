import xarray as xr
import numpy as np
from lidarpy.data.manipulation import molecular_raman_model, z_finder, filter_wavelength
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz


class GetCod:
    """

    https://ntrs.nasa.gov/api/citations/20000092057/downloads/20000092057.pdf

    page: 23

    """
    cod = None
    cod_std = None
    tau = {}
    _mc_bool = True

    def __init__(self, lidar_data: xr.Dataset, cloud_lims: list, lidar_wavelength: int, raman_wavelength: int,
                 angstrom_coeff: float, p_air: np.ndarray, t_air: np.ndarray, pc=True, co2ppmv: int = 392,
                 fit_delta_z=2000, delta_z=100, mc_iter=None):
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
        self.fit_ref = [cloud_lims[0] - fit_delta_z - 100, cloud_lims[0] - 100]
        self.transmittance_ref = z_finder(self.rangebin, cloud_lims[1] + delta_z)
        self.mc_iter = mc_iter

        self.cloud_ref = z_finder(self.rangebin, [cloud_lims[0], cloud_lims[1]])
        alpha = {}
        alpha['elastic_mol'], *_ = AlphaBetaMolecular(p_air,
                                                      t_air,
                                                      lidar_wavelength,
                                                      co2ppmv).get_params()

        alpha['inelastic_mol'], *_ = AlphaBetaMolecular(p_air,
                                                        t_air,
                                                        raman_wavelength,
                                                        co2ppmv).get_params()

        self.tau = (cumtrapz(alpha['elastic_mol'] + alpha['inelastic_mol'],
                             dx=self.rangebin[1] - self.rangebin[0],
                             initial=0)
                    - np.trapz(alpha['elastic_mol'][:self.cloud_ref[0]] + alpha['inelastic_mol'][:self.cloud_ref[0]],
                               dx=self.rangebin[1] - self.rangebin[0]))

        self.scatterer_numerical_density = 78.08e-2 * p_air / (1.380649e-23 * t_air)

    def fit(self):
        if (self.mc_iter is not None) & self._mc_bool:
            return self._mc_fit()
        model_rcs = molecular_raman_model(self.lidar_data, 355, 387, self.p_air, self.t_air, self.fit_ref,
                                          self.co2ppmv, self.pc) * self.rangebin ** 2

        rcs = filter_wavelength(self.lidar_data, self.raman_wavelength, self.pc) * self.rangebin ** 2

        log_ = np.log(
            self.scatterer_numerical_density[self.transmittance_ref:self.transmittance_ref + 150]
            * model_rcs[self.cloud_ref[0]]
            / (self.scatterer_numerical_density[self.cloud_ref[0]]
               * rcs[self.transmittance_ref:self.transmittance_ref + 150])
        )

        tau_cloud = log_ - self.tau[self.transmittance_ref:self.transmittance_ref + 150]

        self.cod = tau_cloud / (1 + (self.lidar_wavelength / self.raman_wavelength) ** self.angstrom_coeff)

        plt.figure(figsize=(12, 5))
        plt.plot(self.rangebin, rcs, "b-", label="RCS 387")
        plt.plot(self.rangebin[self.cloud_ref[1]:self.cloud_ref[1] + 150],
                 rcs[self.cloud_ref[1]:self.cloud_ref[1] + 150], "y*", label="Transmittance")
        fit_b_ref = z_finder(self.rangebin, self.fit_ref)
        plt.plot(self.rangebin[fit_b_ref[0]:fit_b_ref[1]], rcs[fit_b_ref[0]:fit_b_ref[1]], "y--",
                 label="Fit region")
        plt.plot(self.rangebin, model_rcs, "k--", label="base fit")
        plt.yscale("log")
        plt.ylabel("S(z)")
        plt.xlabel("Altitude (m)")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 5))
        z_raman = self.rangebin[self.cloud_ref[1]:self.cloud_ref[1] + 150]
        plt.plot(z_raman, self.cod)
        plt.plot([z_raman[0], z_raman[-1]], [self.cod.mean(), self.cod.mean()], "k--")
        plt.grid()
        plt.title(f"COD mean = {self.cod.mean().round(4)} +- {self.cod.std(ddof=1).round(4)}")
        plt.xlabel("Altitude (m)")
        plt.ylabel("COD raman")
        plt.show()

        return self.cod.mean()

    def _mc_fit(self):
        self._mc_bool = False

        original_ds = self.lidar_data.copy()
        original_inelastic_signal = filter_wavelength(self.lidar_data, self.raman_wavelength, self.pc)
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
