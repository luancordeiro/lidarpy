import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import datetime
from lidarpy.data.manipulation import smooth, smooth_diego_fast, z_finder


def _datevec(ordinal):
    plain_date = datetime.date.fromordinal(int(ordinal))
    date_time = datetime.datetime.combine(plain_date, datetime.datetime.min.time())
    return date_time + datetime.timedelta(days=ordinal - int(ordinal))


class CloudFinder:
    _alt_max = 25000

    def __init__(self, lidar_data: xr.Dataset, sigma, wavelength: int, ref_min: int, window: int, jdz: float,
                 pc: bool = True):
        self._original_data = (lidar_data.phy.sel(wavelength=f"{wavelength}_{int(pc)}")
                               if "wavelength" in lidar_data.dims else lidar_data)
        ref = z_finder(lidar_data.coords["altitude"].data, self._alt_max)
        self.z = lidar_data.coords["altitude"][ref_min:ref].data
        self.signal = self._original_data.phy.data[ref_min:ref]
        self.sigma = sigma[ref_min:ref]
        self.window = window
        self.jdz = jdz

    def _rcs_with_smooth(self):
        signal_w_smooth = smooth(self._original_data.phy.data, self.window)
        z_aux = self._original_data.coords["altitude"].data[::self.window]
        rcs_aux = signal_w_smooth[::self.window] * z_aux ** 2
        f_rcs = interp1d(z_aux, rcs_aux)
        rcs_smooth = f_rcs(self.z)
        rcs_smooth[self.z > 5000] = smooth(rcs_smooth[self.z > 5000], 3)

        rcs_smooth_2 = smooth(rcs_smooth, 71)
        rcs_smooth_3 = rcs_smooth_2.copy()

        return rcs_smooth, rcs_smooth_2, rcs_smooth_3

    def _sigma_rcs(self):
        sigma_rcs_smooth = np.sqrt(smooth((self.sigma / self.window) ** 2, self.window) * self.window) * self.z ** 2
        for alt, coef in zip([7000, 5000, 3000], [1, 3, 1]):
            ref = z_finder(self.z, alt)
            sigma_rcs_smooth[:ref + 1] = coef * sigma_rcs_smooth[ref + 1] + sigma_rcs_smooth[:ref + 1]
        sigma_rcs_smooth_2 = sigma_rcs_smooth.copy()
        return sigma_rcs_smooth, sigma_rcs_smooth_2

    def _get_signal_without_cloud(self, rcs_smooth: np.array, sigma_rcs_smooth_2: np.array, rcs_smooth_sm_2: np.array,
                                  rcs_smooth_sm_3: np.array):
        rcs_smooth_exc = rcs_smooth_sm_2.copy()

        asfend = 25
        init = 1
        for asf in range(asfend + 1):
            rcs_smooth_aux = rcs_smooth.copy()
            npp = 501
            if (asf <= 15) & (asf > init):
                test_z_rcs_smooth_sm = (rcs_smooth_aux - rcs_smooth_sm_2) / sigma_rcs_smooth_2
                mask_aux = test_z_rcs_smooth_sm > 1.5
            else:
                test_z_rcs_smooth_sm = np.abs(rcs_smooth_aux - rcs_smooth_sm_2) / sigma_rcs_smooth_2
                mask_aux = test_z_rcs_smooth_sm > 0.2

            rcs_smooth_aux[mask_aux] = rcs_smooth_sm_2[mask_aux]

            rcs_smooth_sm_2 = smooth(rcs_smooth_aux, npp) if asf != asfend else smooth(rcs_smooth_aux, 71)

            m_aux = rcs_smooth_sm_2 > rcs_smooth_sm_3 + 0.5 * sigma_rcs_smooth_2
            rcs_smooth_sm_2[m_aux] = rcs_smooth_sm_3[m_aux]

            rcs_smooth_exc = rcs_smooth_sm_2.copy()

            if asf == 1:
                p_test = rcs_smooth_exc.copy()
            if asf == asfend - 1:
                r_ts = self.z < 10_000
                rcs_smooth_exc[r_ts] = p_test[r_ts]
                rcs_smooth_sm_2[r_ts] = p_test[r_ts]

        return rcs_smooth_exc

    def _cloud_finder(self, rcs: np.array, sigma_rcs: np.array, rcs_smooth: np.array, rcs_smooth_exc: np.array,
                      sigma_rcs_smooth: np.array) -> tuple:
        snr_exc = rcs_smooth_exc / sigma_rcs
        ind_base, ind_top = [], []
        n1 = 2
        hour = _datevec(self.jdz).hour

        k9101112 = z_finder(self.z, [9000, 10_000, 11_000, 12_000])

        pa2, pd2 = 1, self.window
        rn = pa2 + pd2 + 1

        tz_cond2 = (smooth_diego_fast(rcs - rcs_smooth_exc, pa2, pd2)
                    / ((smooth_diego_fast((sigma_rcs / rn) ** 2, pa2, pd2) * rn) ** .5))

        r = (self.z < 5000) | (self.z > 22_000)
        tz_cond2[r] = 0

        plt.plot(self.z, tz_cond2)
        plt.show()

        k = 2
        cont = 0

        while k <= z_finder(self.z, 20_000):
            k += 1
            cond1 = ((rcs_smooth[k + 0 * self.window] > (rcs_smooth_exc[k + 0 * self.window]
                                                         + n1 * sigma_rcs_smooth[k + 0 * self.window]))
                     & (rcs_smooth[k + 1 * self.window] > (rcs_smooth_exc[k + 1 * self.window]
                                                           + n1 * sigma_rcs_smooth[k + 1 * self.window]))
                     & (rcs_smooth[k + 2 * self.window] > (rcs_smooth_exc[k + 2 * self.window]
                                                           + n1 * sigma_rcs_smooth[k + 2 * self.window])))
            cond2 = (tz_cond2[k] > 4) & (self.z[k] > 10_000) & (snr_exc[k] > 0.01)
            cond = cond1 | cond2

            if cond:
                if (hour <= 6) | (hour >= 18):
                    if (self.z[k] > 10_000) & (sum(snr_exc[k9101112] < 0.1) > 0):
                        continue
                cont += 1
                ind_base.append(k)
                if ind_base[-1] <= 0:
                    ind_base = 1

                for kk in range(k + self.window, len(self.z)):
                    if (rcs_smooth[kk] < rcs_smooth_exc[k]) & (rcs_smooth[kk] < rcs_smooth_exc[kk]):
                        ind_top.append(kk)
                        k = kk+1
                        break

        return ind_base, ind_top

    def _comp(self, rcs_smooth_exc, ind_base, ind_top):
        if (ind_base == []) | (ind_top == []):
            return [np.nan] * 6

        z_base = self.z[ind_base]
        z_top = self.z[ind_top]

        if len(ind_base) > len(ind_top):
            z_base = self.z[ind_base[:-2]]
            ind_base = ind_base[:-1]

        nfz_base = rcs_smooth_exc[ind_base]
        nfz_top = rcs_smooth_exc[ind_top]
        z_max_capa = np.nan
        nfz_max_capa = np.nan

        return z_base, z_top, z_max_capa, nfz_base, nfz_top, nfz_max_capa

    def fit(self):
        rcs_smooth, rcs_smooth_2, rcs_smooth_3 = self._rcs_with_smooth()

        sigma_rcs_smooth, sigma_rcs_smooth_2 = self._sigma_rcs()

        rcs_smooth_exc = self._get_signal_without_cloud(rcs_smooth, sigma_rcs_smooth_2, rcs_smooth_2, rcs_smooth_3)

        ind_base, ind_top = self._cloud_finder(self.signal * self.z ** 2,
                                               self.sigma * self.z ** 2,
                                               rcs_smooth,
                                               rcs_smooth_exc,
                                               sigma_rcs_smooth)

        # plt.plot(self.z / 1e3, rcs, "r-", label="original")
        # plt.plot(self.z / 1e3, rcs_smooth, "k-", label="suavizado")
        # plt.xlabel("Altitude (km)")
        # plt.ylabel("RCS")
        # plt.legend()
        # plt.grid()
        # plt.show()

        # plt.plot(self.z / 1e3, rcs_smooth_exc, "r-", label="rcs_smooth_exc")
        # plt.xlabel("Altitude (km)")
        # plt.ylabel("RCS")
        # plt.legend()
        # plt.grid()
        # plt.show()

        return self._comp(rcs_smooth_exc, ind_base, ind_top)
