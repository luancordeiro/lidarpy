import xarray as xr
import numpy as np
from lidarpy.data.manipulation import molecular_model, z_finder, filter_wavelength
from lidarpy.inversion.klett import Klett
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.interpolate import interp1d


class GetCod:
    _mc_bool = True

    def __init__(self, lidar_data: xr.Dataset, cloud_lims: list, wavelength: int, p_air: np.ndarray, t_air: np.ndarray,
                 pc=True, co2ppmv: int = 392, fit_delta_z=2000, delta_z=200, mc_iter=None):
        self.lidar_data = lidar_data
        self.rangebin = lidar_data.coords["rangebin"].data
        self.wavelength = wavelength
        self.p_air = p_air
        self.t_air = t_air
        self.pc = pc
        self.co2ppmv = co2ppmv
        self.fit_ref = [cloud_lims[0] - fit_delta_z - 100, cloud_lims[0] - 100]
        self.transmittance_ref = z_finder(self.rangebin, cloud_lims[1] + delta_z)
        print(f"ref = {self.transmittance_ref}")
        print(f"z_ref = {self.rangebin[self.transmittance_ref]}")
        self.mc_iter = mc_iter

    def fit(self):
        if (self.mc_iter is not None) & self._mc_bool:
            return self._mc_fit()
        molecular_signal = molecular_model(self.lidar_data,
                                           self.wavelength,
                                           self.p_air,
                                           self.t_air,
                                           self.fit_ref,
                                           self.co2ppmv)

        molecular_rcs = molecular_signal * self.rangebin ** 2

        rcs = filter_wavelength(self.lidar_data, self.wavelength, self.pc) * self.rangebin ** 2

        transmittance_ = (rcs[self.transmittance_ref: self.transmittance_ref + 150]
                          / molecular_rcs[self.transmittance_ref: self.transmittance_ref + 150])

        mean = transmittance_.mean()

        std = transmittance_.std(ddof=1)
        fit_ref = z_finder(self.rangebin, self.fit_ref)
        plt.figure(figsize=(12, 5))
        plt.plot(self.rangebin / 1e3, rcs, "-", color="#0c84a6", alpha=0.8, label="Lidar profile")
        plt.plot(self.rangebin[fit_ref[0]:fit_ref[1]] / 1e3, rcs[fit_ref[0]:fit_ref[1]], "g--", alpha=1,
                 label="Fit region")
        # plt.plot(self.z[cloud_lims], rcs[cloud_lims], "b*", label="Cloud lims")
        plt.plot(self.rangebin / 1e3, molecular_rcs, "k--", label="Mol. profile")
        plt.plot(self.rangebin[self.transmittance_ref: self.transmittance_ref + 150] / 1e3,
                 rcs[self.transmittance_ref: self.transmittance_ref + 150],
                 "g*", alpha=0.4, label="transmittance")
        plt.grid()
        plt.yscale("log")
        plt.legend()
        plt.xlabel("Height (km)")
        plt.ylabel("S(z)")
        plt.tight_layout()
        plt.savefig("figuras/transmittance.jpg", dpi=300)
        plt.show()

        cod_mean = (-0.5 * np.log(transmittance_)).mean()
        cod_std = (-0.5 * np.log(transmittance_)).std(ddof=1) / np.sqrt(len(transmittance_))

        plt.figure(figsize=(12, 5))
        transmittance_z = self.rangebin[self.transmittance_ref: self.transmittance_ref + 150]
        plt.plot(transmittance_z / 1e3, transmittance_, "-", color="#0c84a6", alpha=0.8, linewidth=2,
                 label="Cirrus lidar Transmittance")
        plt.plot([transmittance_z[0] / 1e3, transmittance_z[-1] / 1e3], [mean, mean], "k--",
                 label=r"$T^c_{mean} = $" + str(mean.round(4)))
        plt.title(r"$\tau_{Trans}^c = $" + str(cod_mean.round(3)) + r" $std_{mean} = $" + str(cod_std.round(3)))
        plt.xlabel("Height (km)")
        plt.ylabel(r"$T^c=\exp(-2\tau^c)$")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("figuras/transmittance_tau.jpg", dpi=300)
        plt.show()
        print(cod_mean)
        print(cod_std)

        return cod_mean

    def _mc_fit(self):
        self._mc_bool = False

        original_ds = self.lidar_data.copy()
        original_elastic_signal = filter_wavelength(self.lidar_data, self.wavelength, self.pc)
        elastic_uncertainty = filter_wavelength(self.lidar_data, self.wavelength, self.pc, "sigma")

        elastic_signals = (np.random.randn(self.mc_iter, len(original_elastic_signal)) * elastic_uncertainty
                           + original_elastic_signal)

        self.lidar_data = self.lidar_data.sel(channel=f"{self.wavelength}_{int(self.pc)}")

        taus = []
        for elastic_signal in elastic_signals:
            self.lidar_data.phy.data = elastic_signal
            taus.append(self.fit())

        self.cod = np.mean(taus, axis=0)
        self.cod_std = np.std(taus, ddof=1, axis=0)

        self.lidar_data = original_ds
        self._mc_bool = True

        return self.cod, self.cod_std


def get_lidar_ratio(lidar_data: xr.Dataset, cloud_lims: list, wavelength: int, p_air: np.ndarray, t_air: np.ndarray,
                    z_ref: list, pc: bool = True, co2ppmv: int = 392, correct_noise: bool = True):
    tau_transmittance = GetCod(lidar_data,
                               cloud_lims,
                               wavelength,
                               p_air,
                               t_air,
                               pc,
                               co2ppmv,
                               delta_z=1e3).fit()

    lidar_ratios = np.arange(5, 75, 5)

    cloud_ind = z_finder(lidar_data.coords["rangebin"].data, cloud_lims)

    klett = Klett(lidar_data, wavelength, p_air, t_air, z_ref, 1, pc, co2ppmv, correct_noise)

    taus = []
    for lidar_ratio in lidar_ratios:
        klett.set_lidar_ratio(lidar_ratio)
        alpha, *_ = klett.fit()
        taus.append(trapz(y=alpha[cloud_ind[0]:cloud_ind[1] + 1],
                          dx=lidar_data.coords["rangebin"].data[1] - lidar_data.coords["rangebin"].data[0]))

    difference = (np.array(taus) - tau_transmittance) ** 2

    # print(f"TAU TRANSMITÂNCIA={tau_transmittance}")
    # print(f"LR + TAU KLETT={list(zip(lidar_ratios, np.array(taus)))}")
    # print(f"diferenças quadráticas = {difference}")

    f_diff = interp1d(lidar_ratios, difference, kind="quadratic")

    new_lr = np.linspace(5, 70, 100)

    new_diff = f_diff(new_lr)

    plt.figure(figsize=(12, 5))
    plt.plot(lidar_ratios, difference, "o")
    plt.plot(new_lr, new_diff, "k-", label="Quadratic interpolation")
    plt.plot(new_lr[new_diff.argmin()], min(new_diff), "r*", markersize=10,
             label=r"$LR_{min}$ = " + f"{new_lr[new_diff.argmin()].round(2)} sr")
    # plt.title(r"$LR_{min}$" + f"{new_lr[new_diff.argmin()].round(2)}")
    plt.grid()
    plt.legend()
    plt.xlabel("Lidar ratio (sr)")
    plt.ylabel(r"$(\tau^c_{Klett} - \tau^c_{Trans})^2$")
    plt.tight_layout()
    plt.savefig("figuras/lidar_ratio.jpg")
    plt.show()

    return new_lr[new_diff.argmin()]
