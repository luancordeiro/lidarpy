import numpy as np
import xarray as xr
from math import floor
from scipy.interpolate import interp1d


class CloudFinder:
    def __init__(self, lidar_data: xr.DataArray, sigma, wavelength: int, ref_min: int, window: int, jdz: float,
                 pc: bool):
        self.original_data = lidar_data.copy()
        self.z = lidar_data.coords["altitude"][ref_min:self._find_alt(30_000)]
        self.sigma = sigma
        self.signal = (lidar_data.sel(wavelength=f"{wavelength}_{int(pc)}").data
                       if wavelength in lidar_data.dims else lidar_data.data)[ref_min:self._find_alt(30_000)]
        self.window = window
        self.jdz = jdz

    def _find_alt(self, alt: float) -> int:
        return self.original_data.coords["altitude"].sel(altitude=alt, method="nearest").data

    def _smooth(self, vec: np.array, window: int = None) -> np.array:
        window = self.window if window is None else window
        return vec.copy()

    def _rcs_with_smooth(self):
        signal_w_smooth = self._smooth(self.signal)
        z_aux = self.z[::self.window]
        rcs_aux = signal_w_smooth[::self.window] * z_aux ** 2
        f_rcs = interp1d(z_aux, rcs_aux)
        rcs_smooth = f_rcs(self.z)
        rcs_smooth[self.z > 5000] = self._smooth(rcs_smooth[self.z > 5000], 2)

        rcs_smooth_sm_1 = self._smooth(rcs_smooth, 140)
        rcs_smooth_sm_2 = self._smooth(rcs_smooth, 70)
        rcs_smooth_sm_3 = rcs_smooth_sm_2.copy()

        return rcs_smooth, rcs_smooth_sm_1, rcs_smooth_sm_2, rcs_smooth_sm_3

    def _sigma_rcs(self):
        sigma_rcs_smooth = np.sqrt((self._smooth((self.sigma / self.window) ** 2) * self.window)) * self.z ** 2
        for alt in [7000, 5000, 3000]:
            ref = self._find_alt(alt)
            sigma_rcs_smooth[:ref] = sigma_rcs_smooth[:ref] + sigma_rcs_smooth[ref]
        return sigma_rcs_smooth

    def _range_corrected_signal(self):
        return self.signal * self.z ** 2, self.sigma * self.z ** 2

    def _stat_test(self, rcs_smooth: np.array, sigma_rcs_smooth: np.array, rcs_smooth_sm_2: np.array,
                   rcs_smooth_sm_3: np.array):
        sigma_rcs_smooth_2 = sigma_rcs_smooth.copy()
        asfend = 25
        init = 1
        for asf in range(asfend + 1):
            rcs_smooth_aux = rcs_smooth.copy()
            test_z_rcs_smooth_sm = (rcs_smooth_aux - rcs_smooth_sm_2) / sigma_rcs_smooth_2
            npp = 500
            if (asf <= 15) & (asf > init):
                mask_aux = test_z_rcs_smooth_sm > 1.5
            else:
                mask_aux = test_z_rcs_smooth_sm > 0.2

            rcs_smooth_aux[mask_aux] = rcs_smooth_sm_2[mask_aux]

            rcs_smooth_sm_2 = self._smooth(rcs_smooth_aux, npp) if asf != asfend else self._smooth(rcs_smooth_aux, 70)

            m_aux = rcs_smooth_sm_2 > rcs_smooth_sm_3 + 0.5 * sigma_rcs_smooth_2
            rcs_smooth_sm_2[m_aux] = rcs_smooth_sm_3[m_aux]

            rcs_smooth_exc = rcs_smooth_sm_2.copy()

            if asf == 2:
                p_test = rcs_smooth_exc.copy()
            if asf == asfend - 1:
                r_ts = self.z < 10_000
                rcs_smooth_exc[r_ts] = p_test[r_ts]
                rcs_smooth_sm_2[r_ts] = p_test[r_ts]

        return rcs_smooth_exc

    def fit(self):
        rcs_smooth, rcs_smooth_sm_1, rcs_smooth_sm_2, rcs_smooth_sm_3 = self._rcs_with_smooth()
        sigma_rcs_smooth = self._sigma_rcs()
        rcs, sigma_rcs = self._range_corrected_signal()
        rcs_smooth_exc = self._stat_test(rcs_smooth, sigma_rcs_smooth, rcs_smooth_sm_2, rcs_smooth_sm_3)
