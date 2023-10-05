import numpy as np
import xarray as xr
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz

from lidarpy.utils.functions import z_finder


def remove_background(ds: xr.Dataset, alt_ref: list) -> xr.Dataset:
    """
    Removes the background noise from LIDAR inversion using a reference altitude.

    Args:
        ds (xr.Dataset): LIDAR inversion set to be processed.
        alt_ref (list[int]): Altitude range used to calculate background [start, end].

    Returns:
        xr.Dataset: LIDAR dataset with the background noise removed. The returned dataset
                    also includes background values and their standard deviation.

    Example:
        >>> ds_clean = remove_background(ds, [30_000, 50_000])
    """
    ds_aux = ds.phy.sel(rangebin=slice(*alt_ref))

    background = ds_aux.mean("rangebin")

    background_std = ds_aux.std("rangebin")

    return (
        ds
        .assign(phy=lambda x: x.phy - background)
        .assign(background=background)
        .assign(background_std=background_std)
    )


def groupby_nbins(ds: xr.Dataset, n_bins: int) -> xr.Dataset:
    """
    Groups the LIDAR inversion every n_bins range bins.

    Args:
        ds (xr.Dataset): LIDAR dataset to be grouped.
        n_bins (int): Number of consecutive range bins to group together.

    Returns:
        xr.Dataset: Dataset with observations grouped by the specified number of range bins.

    Note:
        Returns the original dataset unchanged if n_bins is 0 or 1.
    """
    if n_bins in [0, 1]:
        return ds

    rangebin = ds.coords["rangebin"].data

    return (
        ds
        .assign_coords(rangebin=np.arange(len(rangebin)) // n_bins)
        .groupby("rangebin")
        .sum()
        .assign_coords(
            rangebin=lambda x: [rangebin[i * n_bins:(i + 1) * n_bins].mean() for i in range(len(x.rangebin))])
    )


def binshift(ds: xr.Dataset, dead_bin):
    """
    Adjusts inversion bins to account for dead time or other artifacts.

    Args:
        ds (xr.Dataset): LIDAR dataset to be adjusted.
        dead_bin (int): Number of bins to shift.

    Returns:
        xr.Dataset: LIDAR dataset with adjusted bin positions.
    """
    def displace(data_: np.array, bins: int):
        data_ = data_.copy()
        if bins > 0:
            data_[:-bins] = data_[bins:]
        elif bins == 0:
            return data_
        else:
            data_ = np.insert(data_, 0, [0] * (-bins))[:bins]

        return data_

    ds = ds.copy()

    if "channel" not in ds.coords:
        ds.phy.data = displace(ds.phy.data, dead_bin)
        return ds
    dims = None
    new_phys = []
    for channel, dbin in zip(ds.channel.data, dead_bin):
        data = ds.sel(channel=channel).phy.data
        if "time" in ds.coords:
            data_times = []
            for time in ds.time.data:
                data_time = ds.sel(channel=channel, time=time).phy.data
                data_times.append(displace(data_time, dbin))
            new_phys.append(data_times)
            dims = ["channel", "time", "rangebin"]
        else:
            new_phys.append(displace(data, dbin))
            dims = ["channel", "rangebin"]

    ds = ds.assign(phy=xr.DataArray(new_phys, dims=dims))

    return ds


def get_uncertainty(signal: np.array, nshoots: np.array, background: np.array):
    """
    Computes the uncertainty for LIDAR observations.

    Args:
        signal (np.array): Observed LIDAR signal.
        nshoots (np.array): Number of laser shots.
        background (np.array): Background signal values.

    Returns:
        np.array: Uncertainty values corresponding to the input LIDAR signal.
    """
    t = (nshoots / 20e6).reshape(-1, 1)
    n = t * signal * 1e6

    n_bg = t * background * 1e6

    sigma_n = ((n + n_bg) ** 0.5)
    sigma_p = sigma_n * 1e-6 / t

    return sigma_p


def dead_time_correction(lidar_data: xr.Dataset, dead_time: float):
    """
    Applies a dead-time correction to the provided LIDAR dataset.

    Args:
        lidar_data (xr.Dataset): LIDAR dataset to be corrected.
        dead_time (float): Dead-time value for correction.

    Returns:
        xr.Dataset: Corrected LIDAR dataset.
    """
    if "channel" in lidar_data.dims:
        dead_times = np.array([
            dead_time * wavelength.endswith("1") for wavelength in lidar_data.coords["channel"].data
        ]).reshape(-1, 1)
    else:
        dead_times = dead_time
    try:
        new_signals = lidar_data.phy / (1 - dead_times * lidar_data.phy)
    except:
        new_signals = lidar_data.phy / (1 - dead_times.reshape(-1, 1, 1) * lidar_data.phy)

    lidar_data.phy.data = new_signals.data

    return lidar_data


class FindFitRegion:
    min_porc = 0.7
    step = 200

    def __init__(self, signal, sigma, rangebin, molecular_data: xr.Dataset, z_ref):
        self.signal = signal
        self.sigma = sigma
        self.rangebin = rangebin
        self.z_ref = z_ref
        self.model = (molecular_data.beta.data
                      * np.exp(-2 * cumtrapz(rangebin, molecular_data.alpha.data, initial=0)) / rangebin ** 2)

    @staticmethod
    def _chi2_reduced(y_data, y_model, y_err, n_params):
        chi2 = np.sum((y_data - y_model) ** 2 / y_err ** 2)
        dof = len(y_data) - n_params
        return chi2 / dof

    @staticmethod
    def _fitter(x_data, y_data):
        if any(y_data <= 0):
            x_mean = np.mean(x_data)
            x_data = x_data / x_mean

            y_mean = np.mean(y_data)
            y_data = y_data / y_mean
            reg = np.polyfit(x_data, y_data, 1)
            return True, [reg[0] * y_mean / x_mean, reg[1] * x_mean]
        else:
            reg = np.polyfit(np.log(x_data), np.log(y_data), 1)
            return False, reg

    @staticmethod
    def _choose_model(reg, x_data, model_flag):
        """
        Evaluate the model for given regression coefficients and x_data.

        Parameters:
        - reg (array-like): Regression coefficients.
        - x_data (array-like): Independent variable inversion.
        - model_flag (bool): Model type flag.

        Returns:
        - array-like: Model output.
        """
        if model_flag:
            return reg[0] * x_data + reg[1]
        else:
            return np.exp(reg[0] * np.log(x_data) + reg[1])

    def _find_best_region(self, regions):
        """
        Finds the best region with the minimum reduced chi-squared value.

        Parameters:
        - x_data, y_data, y_err (array-like): Data arrays.
        - regions (list): List of candidate regions.

        Returns:
        - list: Region with the minimum reduced chi-squared value.
        """

        min_chi2_reduced = 999999999
        min_region = -4

        chis2 = []
        regs = []
        flags = []
        for i, region in enumerate(regions):
            x_fit = self.model[region[0]:region[1]]
            y_fit = savgol_filter(self.signal, 11, 2)
            j = 0
            while any(y_fit[region[0]:region[1]] <= 0) & (j <= 5):
                y_fit = savgol_filter(y_fit, 11, 2)
                j += 1
            y_fit = y_fit[region[0]:region[1]]
            model_flag, reg = self._fitter(x_fit, y_fit)
            chi2_reduced_current = self._chi2_reduced(self.signal[region[0]:region[1]],
                                                      self._choose_model(reg, self.model[region[0]:region[1]],
                                                                         model_flag),
                                                      self.sigma[region[0]:region[1]],
                                                      len(reg))

            chis2.append(chi2_reduced_current)
            regs.append(reg)
            flags.append(model_flag)

            if chi2_reduced_current <= min_chi2_reduced * self.min_porc:
                min_chi2_reduced = chi2_reduced_current
                min_region = i
        print(chis2)
        return regions[min_region]

    def set_min_chi2_reduction(self, min_porc):
        self.min_porc = min_porc
        return self

    def set_reference_step(self, step):
        self.step = step
        return self

    def fit(self):
        references = [[alt_bot, self.z_ref[1]]
                      for alt_bot in range(int(self.z_ref[0]), int(self.z_ref[1] - self.step * 2), self.step)]
        print(references)
        references = [z_finder(self.rangebin, ref_) for ref_ in references]
        ref = self._find_best_region(references)

        return ref
