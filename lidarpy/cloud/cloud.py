import numpy as np
from scipy.interpolate import interp1d
import datetime
from lidarpy.utils.functions import smooth, smooth_diego_fast, z_finder


class CloudFinder:
    """
    CloudFinder identifies cloud layers in LIDAR signal inversion.

    Analyzes LIDAR signals to detect cloud layers based on specific conditions and statistical measures.

    Attributes:
        _Z_MAX (int): Maximum altitude in meters.
        _original_data (np.array): Original LIDAR signal inversion.
        ref_min (int): Minimum reference altitude index.
        ref_max (int): Maximum reference altitude index.
        z (np.array): Altitude values derived from LIDAR inversion.
        signal (np.array): LIDAR signal inversion subset for reference altitudes.
        sigma (np.array): Standard deviation of the LIDAR signal inversion within reference altitudes.
        window (int): Window size for smoothing operations.
        datetime (float): Julian date representation.
    """
    _Z_MAX = 25_000

    def __init__(self, rangebin: np.array, signal: np.array, sigma: np.array, z_min: float, window: int,
                 time: datetime):
        self._original_data = signal.copy()
        self.ref_min, self.ref_max = z_finder(rangebin, z_min), z_finder(rangebin, self._Z_MAX)
        self.signal = signal[self.ref_min:self.ref_max]
        self.z = rangebin[self.ref_min:self.ref_max]
        self.sigma = sigma[self.ref_min:self.ref_max]
        self._sigma_original = sigma.copy()
        self.window = window
        self.datetime = time

    def _rcs_with_smooth(self):
        signal_w_smooth = smooth(self.signal, self.window)[::self.window]

        z_aux = self.z[::self.window]
        rcs_aux = signal_w_smooth * z_aux ** 2

        if z_aux[-1] != self.z[-1]:
            z_aux = np.append(z_aux, self.z[-1])
            rcs_aux = np.append(rcs_aux, signal_w_smooth[-1] * self.z[-1] ** 2)

        f_rcs = interp1d(z_aux, rcs_aux)
        rcs_smooth = f_rcs(self.z)
        rcs_smooth[-1] = 0
        rcs_smooth[self.z > 5000] = smooth(rcs_smooth[self.z > 5000], 1)

        rcs_smooth_2 = smooth(rcs_smooth, 69)
        rcs_smooth_3 = rcs_smooth_2.copy()

        return rcs_smooth, rcs_smooth_2, rcs_smooth_3

    def _sigma_rcs(self):
        sigma2 = self._sigma_original.copy()
        aux = np.unique(sigma2)
        aux.sort()
        if len(aux) > 1:
            sigma2[sigma2 <= aux[0]] = aux[1]

        sigma_rcs_smooth = ((smooth((sigma2[self.ref_min:self.ref_max] / self.window) ** 2, self.window)
                             * self.window) ** 0.5) * self.z ** 2

        for alt, coef in zip([7000, 5000, 3000], [1, 3, 1]):
            ref = z_finder(self.z, alt)
            sigma_rcs_smooth[:ref + 1] = coef * sigma_rcs_smooth[ref + 1] + sigma_rcs_smooth[:ref + 1]

        sigma_rcs_smooth_2 = sigma_rcs_smooth.copy()

        return sigma_rcs_smooth, sigma_rcs_smooth_2

    def _get_signal_without_cloud(self, rcs_smooth: np.array, sigma_rcs_smooth_2: np.array, rcs_smooth_sm_2: np.array,
                                  rcs_smooth_sm_3: np.array):
        rcs_smooth_exc = rcs_smooth_sm_2.copy()

        p_test = None
        asfend = 26
        for asf in range(1, asfend):
            rcs_smooth_aux = rcs_smooth.copy()
            npp = 499
            if (asf <= 15) & (asf > 1):
                test_z_rcs_smooth_sm = (rcs_smooth_aux - rcs_smooth_sm_2) / sigma_rcs_smooth_2
                mask_aux = test_z_rcs_smooth_sm > 1.5
            else:
                test_z_rcs_smooth_sm = np.abs(rcs_smooth_aux - rcs_smooth_sm_2) / sigma_rcs_smooth_2
                mask_aux = test_z_rcs_smooth_sm > 0.2

            rcs_smooth_aux[mask_aux] = rcs_smooth_sm_2[mask_aux]

            rcs_smooth_sm_2 = smooth(rcs_smooth_aux, npp) if asf != (asfend - 1) else smooth(rcs_smooth_aux, 69)

            m_aux = rcs_smooth_sm_2 > rcs_smooth_sm_3 + 0.5 * sigma_rcs_smooth_2
            rcs_smooth_sm_2[m_aux] = rcs_smooth_sm_3[m_aux]

            rcs_smooth_exc = rcs_smooth_sm_2.copy()

            if asf < 3:
                rcs_smooth_exc = rcs_smooth_exc.copy()
            else:
                r_ts = self.z > 8500
                rcs_smooth_exc[r_ts] = rcs_smooth_exc[r_ts]
            if asf == 2:
                p_test = rcs_smooth_exc.copy()
            if asf == asfend - 2:
                r_ts = self.z < 10_000
                rcs_smooth_exc[r_ts] = p_test[r_ts]
                rcs_smooth_sm_2[r_ts] = p_test[r_ts]

        return rcs_smooth_exc

    def _cloud_finder(self, rcs: np.array, sigma_rcs: np.array, rcs_smooth: np.array, rcs_smooth_exc: np.array,
                      sigma_rcs_smooth: np.array) -> tuple:
        snr_exc = rcs_smooth_exc / sigma_rcs
        ind_base, ind_top = [], []
        n1 = 2
        hour = self.datetime.hour

        k9101112 = z_finder(self.z, [9000, 10_000, 11_000, 12_000])

        pa2, pd2 = 1, self.window
        rn = pa2 + pd2 + 1

        tz_cond2 = (smooth_diego_fast(rcs - rcs_smooth_exc, pa2, pd2)
                    / ((smooth_diego_fast((sigma_rcs / rn) ** 2, pa2, pd2) * rn) ** .5))

        r = (self.z < 5000) | (self.z > 22_000)
        tz_cond2[r] = 0

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
                        k = kk + 1
                        break

        return ind_base, ind_top

    def _distance_between_clouds_correction(self):
        pass
        if len(self.z_base) > 1:
            ind_clear_zt = np.array(np.where((self.z_base[1:] - self.z_top[:-1]) < 500)[0])
            ind_clear_zb = ind_clear_zt + 1
            self.z_top = np.delete(self.z_top, ind_clear_zt)
            self.z_base = np.delete(self.z_base, ind_clear_zb)

    def _comp(self, rcs_smooth_exc, ind_base, ind_top):
        if (ind_base == []) | (ind_top == []):
            self.z_base, self.z_top, self.z_max_capa, self.nfz_base, self.nfz_top, self.nfz_max_capa = [[np.nan]] * 6
            return self.z_base, self.z_top, self.z_max_capa, self.nfz_base, self.nfz_top, self.nfz_max_capa

        z_base = self.z[ind_base]
        z_top = self.z[ind_top]

        self.z_base, self.z_top = np.array(z_base), np.array(z_top)

        self._distance_between_clouds_correction()

        new_ind_base = z_finder(self.z, self.z_base)
        new_ind_top = z_finder(self.z, self.z_top)

        nfz_base = rcs_smooth_exc[new_ind_base]
        nfz_top = rcs_smooth_exc[new_ind_top]
        nfz_max_capa = []
        z_max_capa = []
        for base, top in zip(new_ind_base, new_ind_top):
            max_signal = max(self.signal[base:top])
            bools = self.signal[base:top] == max_signal
            z_max = self.z[base:top][np.where(bools)[0][0]]
            nfz_max_capa.append(max_signal * z_max ** 2)
            z_max_capa.append(z_max)

        self.z_max_capa, self.nfz_base, self.nfz_top, self.nfz_max_capa = (np.array(z_max_capa),
                                                                           np.array(nfz_base),
                                                                           np.array(nfz_top),
                                                                           np.array(nfz_max_capa))

        return self.z_base, self.z_top, self.z_max_capa, self.nfz_base, self.nfz_top, self.nfz_max_capa

    def fit(self):
        rcs_smooth, rcs_smooth_2, rcs_smooth_3 = self._rcs_with_smooth()

        sigma_rcs_smooth, sigma_rcs_smooth_2 = self._sigma_rcs()

        rcs_smooth_exc = self._get_signal_without_cloud(rcs_smooth, sigma_rcs_smooth_2, rcs_smooth_2, rcs_smooth_3)

        ind_base, ind_top = self._cloud_finder(self.signal * self.z ** 2,
                                               self.sigma * self.z ** 2,
                                               rcs_smooth,
                                               rcs_smooth_exc,
                                               sigma_rcs_smooth)

        return self._comp(rcs_smooth_exc, ind_base, ind_top)


def cloud_mask_correction(z_bases, z_tops, z_max_capas, nfz_bases, nfz_tops, nfz_max_capas):
    z_bases_, z_tops_, z_max_capas_, nfz_bases_, nfz_tops_, nfz_max_capas_ = (z_bases.copy(), z_tops.copy(),
                                                                              z_max_capas.copy(), nfz_bases.copy(),
                                                                              nfz_tops.copy(), nfz_max_capas.copy())
    """
    Corrects cloud masks based on proximity criteria between different layers of cloud inversion.

    Parameters:
    - z_bases (list of lists of floats): Base heights of the clouds.
    - z_tops (list of lists of floats): Top heights of the clouds.
    - z_max_capas (list of lists of floats): Maximum heights within the cloud layers.
    - nfz_bases (list of lists of floats): Base heights of the no-fly zones.
    - nfz_tops (list of lists of floats): Top heights of the no-fly zones.
    - nfz_max_capas (list of lists of floats): Maximum heights within the no-fly zone layers.

    Returns:
    - tuple of lists: Corrected versions of z_bases, z_tops, z_max_capas, nfz_bases, nfz_tops, and nfz_max_capas.

    Notes:
    The function works by iterating over the cloud inversion and checking for nearby layers within specified thresholds.
    Layers that match the criteria are considered neighbors and are used in the correction process. The function also 
    considers a temporal component in the checking process, allowing for comparison of layers from nearby time steps.

    The function makes use of several hard-coded parameters:
    - dt: Temporal proximity for the comparison.
    - dz1 and dz2: Vertical proximity thresholds based on the thickness of the layer.
    - dthick: Threshold for determining which dz value to use.
    """
    dt = 1
    dz1 = 500
    dz2 = 300
    dthick = 1e3
    for u, (bases, tops) in enumerate(zip(z_bases, z_tops)):
        if np.isnan(bases)[0]:
            continue
        neib = np.zeros(len(bases), dtype=bool)
        for i, (base, top) in enumerate(zip(bases, tops)):
            uaux = list(range(u - dt, u + dt + 1))
            for uj in uaux:
                if (uj < 0) | (uj == len(z_bases)) | (uj == u):
                    continue
                elif (np.isnan(z_bases[uj]).sum() > 0) | (np.isnan(z_tops[uj]).sum() > 0):
                    continue
                elif top - base > dthick:
                    dz = dz1
                    cond1 = sum(abs(base - z_bases[uj]) < dz) > 0
                    cond2 = sum(abs(base - z_tops[uj]) < dz) > 0
                    cond3 = sum(abs(top - z_bases[uj]) < dz) > 0
                    cond4 = sum(abs(base - z_tops[uj]) < dz) > 0
                    if cond1 | cond2 | cond3 | cond4:
                        neib[i] = True
                else:
                    dz = dz2
                    cond1 = sum(abs(base - z_bases[uj]) < dz) > 0
                    cond2 = sum(abs(base - z_tops[uj]) < dz) > 0
                    cond3 = sum(abs(top - z_bases[uj]) < dz) > 0
                    cond4 = sum(abs(base - z_tops[uj]) < dz) > 0
                    if cond1 | cond2 | cond3 | cond4:
                        neib[i] = True

        z_bases_[u] = ([np.nan] if z_bases[u][neib].size == 0 else z_bases[u][neib])
        z_tops_[u] = ([np.nan] if z_tops[u][neib].size == 0 else z_tops[u][neib])
        z_max_capas_[u] = ([np.nan] if z_max_capas[u][neib].size == 0 else z_max_capas[u][neib])
        nfz_bases_[u] = ([np.nan] if nfz_bases[u][neib].size == 0 else nfz_bases[u][neib])
        nfz_tops_[u] = ([np.nan] if nfz_tops[u][neib].size == 0 else nfz_tops[u][neib])
        nfz_max_capas_[u] = ([np.nan] if nfz_max_capas[u][neib].size == 0 else nfz_max_capas[u][neib])

    return z_bases_, z_tops_, z_max_capas_, nfz_bases_, nfz_tops_, nfz_max_capas_
