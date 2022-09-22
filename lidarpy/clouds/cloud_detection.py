import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp1d
import datetime


def _datevec(ordinal):
    plain_date = datetime.date.fromordinal(int(ordinal))
    date_time = datetime.datetime.combine(plain_date, datetime.datetime.min.time())
    return date_time + datetime.timedelta(days=ordinal - int(ordinal))


class CloudFinder:
    _alt_max = 25000

    def __init__(self, lidar_data: xr.DataArray, sigma, wavelength: int, ref_min: int, window: int, jdz: float,
                 pc: bool = True):
        self._original_data = (lidar_data.sel(wavelength=f"{wavelength}_{int(pc)}")
                               if "wavelength" in lidar_data.dims else lidar_data)
        alt = self._original_data.coords["altitude"].sel(altitude=self._alt_max, method="nearest").data
        ref = np.where(self._original_data.coords["altitude"] == alt)[0][0]
        self.z = lidar_data.coords["altitude"][ref_min:ref].data
        self.sigma = sigma[ref_min:ref]
        self.signal = self._original_data.data[ref_min:ref]
        self.window = window
        self.jdz = jdz

    def _find_alt(self, alt: float) -> int:
        return np.where(abs(self.z - alt) == min(abs(self.z - alt)))[0][0]

    def _smooth(self, vec: np.array, window: int = None) -> np.array:
        window = self.window if window is None else window
        return pd.Series(vec).rolling(window, min_periods=1).mean().to_numpy()

    def _rcs_with_smooth(self):
        signal_w_smooth = self._smooth(self._original_data.data)
        z_aux = self._original_data.coords["altitude"].data[::self.window]
        rcs_aux = signal_w_smooth[::self.window] * z_aux ** 2
        f_rcs = interp1d(z_aux, rcs_aux)
        print(len(z_aux))
        print(len(self.z))
        print(z_aux[-1])
        print(self.z[-1])
        rcs_smooth = f_rcs(self.z)
        rcs_smooth[self.z > 5000] = self._smooth(rcs_smooth[self.z > 5000], 2)

        rcs_smooth_2 = self._smooth(rcs_smooth, 70)
        rcs_smooth_3 = rcs_smooth_2.copy()

        return rcs_smooth, rcs_smooth_2, rcs_smooth_3

    def _sigma_rcs(self):
        sigma_rcs_smooth = np.sqrt(self._smooth((self.sigma / self.window) ** 2) * self.window) * self.z ** 2
        sigma_rcs_smooth_2 = sigma_rcs_smooth.copy()
        for alt, coef in zip([7000, 5000, 3000], [1, 3, 1]):
            ref = self._find_alt(alt)
            sigma_rcs_smooth[:ref + 1] = coef * sigma_rcs_smooth[ref + 1] + sigma_rcs_smooth[:ref + 1]
        return sigma_rcs_smooth, sigma_rcs_smooth_2

    def _stat_test(self, rcs_smooth: np.array, sigma_rcs_smooth_2: np.array, rcs_smooth_sm_2: np.array,
                   rcs_smooth_sm_3: np.array):
        rcs_smooth_exc = rcs_smooth_sm_2.copy()

        asfend = 25
        init = 1
        plt.figure()
        for asf in range(asfend + 1):
            rcs_smooth_aux = rcs_smooth.copy()
            npp = 500
            if (asf <= 15) & (asf > init):
                test_z_rcs_smooth_sm = (rcs_smooth_aux - rcs_smooth_sm_2) / sigma_rcs_smooth_2
                mask_aux = test_z_rcs_smooth_sm > 1.5
            else:
                test_z_rcs_smooth_sm = np.abs(rcs_smooth_aux - rcs_smooth_sm_2) / sigma_rcs_smooth_2
                mask_aux = test_z_rcs_smooth_sm > 0.2

            rcs_smooth_aux[mask_aux] = rcs_smooth_sm_2[mask_aux]

            rcs_smooth_sm_2 = self._smooth(rcs_smooth_aux, npp) if asf != asfend else self._smooth(rcs_smooth_aux, 70)

            m_aux = rcs_smooth_sm_2 > rcs_smooth_sm_3 + 0.5 * sigma_rcs_smooth_2
            rcs_smooth_sm_2[m_aux] = rcs_smooth_sm_3[m_aux]

            rcs_smooth_exc = rcs_smooth_sm_2.copy()

            plt.plot(self.z, rcs_smooth_exc)

            if asf == 1:
                p_test = rcs_smooth_exc.copy()
            if asf == asfend - 1:
                r_ts = self.z < 10_000
                rcs_smooth_exc[r_ts] = p_test[r_ts]
                rcs_smooth_sm_2[r_ts] = p_test[r_ts]
        plt.show()

        return rcs_smooth_exc

    def _cloud_finder(self, rcs: np.array, sigma_rcs: np.array, rcs_smooth: np.array, rcs_smooth_exc: np.array,
                      sigma_rcs_smooth: np.array) -> tuple:
        snr_exc = rcs_smooth_exc / sigma_rcs
        ind_base, ind_top = [], []
        n1 = 2
        hour = _datevec(self.jdz).hour

        k9101112 = [self._find_alt(alt) for alt in [9000, 10_000, 11_000, 12_000]]

        pa2, pd2 = 1, self.window
        rn = pa2 + pd2 + 1

        tz_cond2 = (self._smooth_diego_fast(rcs - rcs_smooth_exc, pa2, pd2)
                    / ((self._smooth_diego_fast((sigma_rcs / rn) ** 2, pa2, pd2) * rn) ** 5))

        r = (self.z < 5000) | (self.z > 22_000)
        tz_cond2[r] = 0
        for cont, k in enumerate(range(3, self._find_alt(20_000))):
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
                ind_base.append(k - round(0 * self.window))
                if ind_base[-1] <= 0:
                    ind_base = 1

                for kk in range(k + self.window, len(self.z)):
                    if (rcs_smooth[kk] < rcs_smooth_exc[k]) & (rcs_smooth[kk] < rcs_smooth_exc[kk]):
                        ind_top.append(kk)
                        break

        return ind_base, ind_top

    def _smooth_diego_fast(self, y, p_before, p_after):
        sm = p_before + p_after + 1
        y_sm = self._smooth(y, sm)
        y_sm4 = np.zeros(y_sm.shape)
        y_sm4[:-sm//2 + 1] = y_sm[sm // 2:]
        return y_sm4

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

        # for ib, it in zip(ind_base, ind_top):
        #     max_ = max(self.signal[ib:it])
        #     ind = np.where(self.signal == max_)[0][0]
        #     z_max_capa = self.z[ind]
        #     print("arrumar o nfz_max_capa")
        #     nfz_max_capa = np.nan

        return z_base, z_top, z_max_capa, nfz_base, nfz_top, nfz_max_capa

    def fit(self):
        rcs_smooth, rcs_smooth_2, rcs_smooth_3 = self._rcs_with_smooth()
        sigma_rcs_smooth, sigma_rcs_smooth_2 = self._sigma_rcs()
        rcs, sigma_rcs = self.signal * self.z ** 2, self.sigma * self.z ** 2

        plt.plot(self.z / 1e3, rcs, "r-", label="original")
        plt.plot(self.z / 1e3, rcs_smooth, "k-", label="suavizado")
        plt.xlabel("Altitude (km)")
        plt.ylabel("RCS")
        plt.legend()
        plt.grid()
        plt.show()

        rcs_smooth_exc = self._stat_test(rcs_smooth, sigma_rcs_smooth_2, rcs_smooth_2, rcs_smooth_3)

        plt.plot(self.z / 1e3, rcs_smooth_exc, "r-", label="rcs_smooth_exc")
        plt.xlabel("Altitude (km)")
        plt.ylabel("RCS")
        plt.legend()
        plt.grid()
        plt.show()

        ind_base, ind_top = self._cloud_finder(rcs, sigma_rcs, rcs_smooth, rcs_smooth_exc, sigma_rcs_smooth)
        return self._comp(rcs_smooth_exc, ind_base, ind_top)
