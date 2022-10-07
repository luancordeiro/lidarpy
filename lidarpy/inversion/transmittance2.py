import xarray as xr
import numpy as np
from scipy.integrate import trapz
from lidarpy.data.manipulation import molecular_model
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Transmittance:
    _alpha = None
    _beta = None
    tau = None

    def __init__(self,
                 lidar_data: xr.DataArray,
                 cloud_lims: list,
                 wavelength: int,
                 p_air: np.ndarray,
                 t_air: np.ndarray,
                 pc=True,
                 co2ppmv: int = 392,
                 delta_z=200):
        if "wavelength" in lidar_data.dims:
            self.signal = lidar_data.sel(wavelength=f"{wavelength}_{int(pc)}").data
        else:
            self.signal = lidar_data.data

        self.z = lidar_data.coords["altitude"].data

        fit_ref = [cloud_lims[0] - 2000, cloud_lims[0]]

        molecular_signal = molecular_model(lidar_data,
                                           wavelength,
                                           p_air,
                                           t_air,
                                           fit_ref,
                                           co2ppmv)

        molecular_rcs = molecular_signal * self.z ** 2

        transmittance_ref = cloud_lims[1] + delta_z
        transmittance_ref = lidar_data.coords["altitude"].sel(altitude=transmittance_ref, method="nearest").data
        transmittance_ref = np.where(self.z == transmittance_ref)[0][0]

        rcs = self.signal * self.z ** 2

        plt.figure(figsize=(12, 5))
        plt.plot(self.z, rcs, "b-", label="Lidar profile")
        plt.plot(self.z[fit_ref[0]:fit_ref[1]], rcs[fit_ref[0]:fit_ref[1]], "y--", label="Fit region")
        # plt.plot(self.z[cloud_lims], rcs[cloud_lims], "b*", label="Cloud lims")
        plt.plot(self.z, molecular_rcs, "k-", label="Mol. profile")
        plt.plot(self.z[transmittance_ref: transmittance_ref + 150], rcs[transmittance_ref: transmittance_ref + 150],
                 "y*", label="transmittance")
        plt.grid()
        plt.yscale("log")

        plt.legend()
        plt.xlabel("altitude (m)")
        plt.ylabel("S(z)")

        plt.show()

        transmittance_z = self.z[transmittance_ref: transmittance_ref + 150]
        transmittance = rcs[transmittance_ref: transmittance_ref + 150] / molecular_rcs[
                                                                          transmittance_ref: transmittance_ref + 150]

        mean = transmittance.mean()
        std = transmittance.std(ddof=1)

        plt.figure(figsize=(12, 5))
        plt.plot(transmittance_z, transmittance)
        plt.plot([transmittance_z[0], transmittance_z[-1]], [mean, mean], "k--")
        plt.title(f"mean value = {mean}")
        plt.xlabel("altitude (m)")
        plt.ylabel("Transmittance")
        plt.show()

        print("AOD =", -0.5 * np.log(mean), "+-",
              -0.5 * np.log(transmittance.std(ddof=1)) / np.sqrt(len(transmittance)))

        self.tau = -0.5 * np.log(mean)

    def fit(self):
        return self.tau

