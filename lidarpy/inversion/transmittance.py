import xarray as xr
import numpy as np
from lidarpy.data.manipulation import molecular_model, z_finder, filter_wavelength
from lidarpy.inversion.klett import Klett
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.interpolate import interp1d


def get_cod(lidar_data: xr.Dataset, cloud_lims: list, wavelength: int, p_air: np.ndarray, t_air: np.ndarray,
            pc=True, co2ppmv: int = 392, delta_z=200):
    signal = filter_wavelength(lidar_data, wavelength, pc)

    z = lidar_data.coords["altitude"].data

    fit_ref = [cloud_lims[0] - 2000, cloud_lims[0]]

    transmittance_ref = z_finder(lidar_data.coords["altitude"].data, cloud_lims[1] + delta_z)

    molecular_signal = molecular_model(lidar_data,
                                       wavelength,
                                       p_air,
                                       t_air,
                                       fit_ref,
                                       co2ppmv)

    molecular_rcs = molecular_signal * z ** 2

    rcs = signal * z ** 2

    transmittance_ = (rcs[transmittance_ref: transmittance_ref + 150]
                      / molecular_rcs[transmittance_ref: transmittance_ref + 150])

    mean = transmittance_.mean()

    std = transmittance_.std(ddof=1)

    plt.figure(figsize=(12, 5))
    plt.plot(z, rcs, "b-", label="Lidar profile")
    plt.plot(z[fit_ref[0]:fit_ref[1]], rcs[fit_ref[0]:fit_ref[1]], "y--", label="Fit region")
    # plt.plot(self.z[cloud_lims], rcs[cloud_lims], "b*", label="Cloud lims")
    plt.plot(z, molecular_rcs, "k-", label="Mol. profile")
    plt.plot(z[transmittance_ref: transmittance_ref + 150], rcs[transmittance_ref: transmittance_ref + 150],
             "y*", label="transmittance")
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.xlabel("altitude (m)")
    plt.ylabel("S(z)")
    plt.show()

    plt.figure(figsize=(12, 5))
    transmittance_z = z[transmittance_ref: transmittance_ref + 150]
    plt.plot(transmittance_z, transmittance_)
    plt.plot([transmittance_z[0], transmittance_z[-1]], [mean, mean], "k--")
    plt.title(f"mean value = {mean.round(4)} +- {std.round(4)}")
    plt.xlabel("altitude (m)")
    plt.ylabel("Transmittance")
    plt.show()

    return -0.5 * np.log(mean)


def get_lidar_ratio(lidar_data: xr.Dataset, cloud_lims: list, wavelength: int, p_air: np.ndarray, t_air: np.ndarray,
                    z_ref: list, pc: bool = True, co2ppmv: int = 392, correct_noise: bool = True):
    tau_transmittance = get_cod(lidar_data,
                                cloud_lims,
                                wavelength,
                                p_air,
                                t_air,
                                pc,
                                co2ppmv)

    lidar_ratios = np.arange(5, 75, 5)

    cloud_ind = z_finder(lidar_data.coords["altitude"].data, cloud_lims)

    klett = Klett(lidar_data, wavelength, p_air, t_air, z_ref, 1, pc, co2ppmv, correct_noise)

    taus = []
    for lidar_ratio in lidar_ratios:
        klett.set_lidar_ratio(lidar_ratio)
        alpha, *_ = klett.fit()
        taus.append(trapz(alpha[cloud_ind[0]:cloud_ind[1] + 1],
                          lidar_data.coords["altitude"].data[cloud_ind[0]:cloud_ind[1] + 1]))

    difference = (np.array(taus) - tau_transmittance) ** 2

    f_diff = interp1d(lidar_ratios, difference, kind="quadratic")

    new_lr = np.linspace(5, 70, 100)

    new_diff = f_diff(new_lr)

    plt.plot(lidar_ratios, difference, "o")
    plt.plot(new_lr, new_diff, "k-")
    plt.plot(new_lr[new_diff.argmin()], min(new_diff), "*")
    plt.title(f"lidar ratio = {new_lr[new_diff.argmin()].round(2)}")
    plt.grid()
    plt.xlabel("Lidar ratio")
    plt.ylabel("Difference")
    plt.show()

    return new_lr[new_diff.argmin()]
