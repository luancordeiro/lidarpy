import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import datetime as dt


def matlab2datetime(matlab_datenum):
    """
        Converts a MATLAB datenum into a Python datetime object.

        Parameters:
        -----------
        matlab_datenum : float
            The MATLAB datenum to be converted.

        Returns:
        --------
        datetime.datetime
            Corresponding Python datetime object.

        Note:
        -----
        MATLAB's datenum starts from year 0, whereas Python's datetime starts from year 1. This function
        adjusts for that difference.
    """
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days=366)
    return day + dayfrac


# def molecular_raman_model(signal: np.array, rangebin: np.array, lidar_wavelength, raman_wavelength, p_air, t_air,
# alt_ref, co2ppmv=375) -> np.array:
#     """
#     Calculate the molecular Raman model.
#
#     Parameters:
#     - lidar_data: Input lidar dataset.
#     - lidar_wavelength: Lidar operating wavelength.
#     - raman_wavelength: Raman scattering wavelength.
#     - p_air: Air pressure.
#     - t_air: Air temperature.
#     - alt_ref: Altitude reference.
#     - co2ppmv: Carbon dioxide concentration in ppmv. Default is 375.
#     - pc: Photocount mode. Default is True.
#
#     Returns:
#     - np.array: Molecular Raman model.
#     """
#     alpha_lidar_mol, *_ = AlphaBetaMolecular(p_air, t_air, lidar_wavelength, co2ppmv).get_params()
#
#     alpha_raman_mol, *_ = AlphaBetaMolecular(p_air, t_air, raman_wavelength, co2ppmv).get_params()
#
#     scatterer_numerical_density = 78.08e-2 * p_air / (1.380649e-23 * t_air)
#
#     model = (scatterer_numerical_density * np.exp(-cumulative_trapezoid(rangebin, alpha_lidar_mol + alpha_raman_mol, initial=0))
#              / rangebin ** 2)
#
#     ref = z_finder(rangebin, alt_ref)
#
#     model_fit = np.log(model[ref[0]:ref[1]])
#
#     signal_fit = np.log(signal[ref[0]:ref[1]])
#
#     nans = (np.isnan(signal_fit) == False)
#
#     reg = np.polyfit(model_fit[nans], signal_fit[nans], 1)
#
#     return np.exp(reg[0] * np.log(model) + reg[1])


def smooth(signal: np.array, window: int):
    """
    Smooth a given signal using a moving average.

    Parameters:
    - signal: Input signal array.
    - window: Window size for smoothing.

    Returns:
    - np.array: Smoothed signal.
    """
    if window % 2 == 0:
        raise Exception("Window value must be odd")
    out0 = np.convolve(signal, np.ones(window, dtype=int), 'valid') / window
    r = np.arange(1, window - 1, 2)
    start = np.cumsum(signal[:window - 1])[::2] / r
    stop = (np.cumsum(signal[:-window:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def smooth_diego_fast(signal: np.array, p_before: int, p_after: int):
    """
    Fast smoothing method implemented by Gouveia 2018.

    Parameters:
    - signal: Input signal array.
    - p_before: Points before.
    - p_after: Points after.

    Returns:
    - np.array: Smoothed signal.
    """
    sm = p_before + p_after + 1
    y_sm = smooth(signal, sm)
    y_sm4 = np.zeros(y_sm.shape)
    y_sm4[:-sm // 2 + 1] = y_sm[sm // 2:]
    return y_sm4


def signal_smoother(rangebin: np.array, signal: np.array, window: int):
    """
    Smooth a signal.

    Parameters:
    - signal: Input signal array.
    - rangebin: Range bin array.
    - window: Window size for smoothing.

    Returns:
    - np.array: Smoothed signal.
    """
    vec_smooth = smooth(signal, window)
    vec_aux = vec_smooth[::window]
    z_aux = rangebin[::window]
    if z_aux[-1] < rangebin[-1]:
        ind = z_finder(rangebin, z_aux[-1])
        vec_aux = np.append(vec_aux, signal[ind:])
        z_aux = np.append(z_aux, rangebin[ind:])
    func = interp1d(z_aux, vec_aux)

    return func(rangebin)


def z_finder(rangebin: np.array, alts):
    """
    Find the indices of altitudes in the rangebin.

    Parameters:
    - rangebin: Range bin array.
    - alts: Desired altitudes.

    Returns:
    - Indices corresponding to the given altitudes.
    """
    def finder(z: int):
        return round((z - rangebin[0]) / (rangebin[1] - rangebin[0]))

    try:
        iter(alts)
        iterable = True
    except TypeError:
        iterable = False

    if alts is None:
        return None
    elif iterable:
        return [finder(alt) for alt in alts]
    else:
        return finder(alts)


def molecular_model(rangebin, signal, molecular_data, alt_ref) -> np.array:
    alpha_mol, beta_mol = molecular_data.alpha.data, molecular_data.beta.data

    model = beta_mol * np.exp(-2 * cumulative_trapezoid(rangebin, alpha_mol, initial=0)) / rangebin ** 2

    ref = z_finder(rangebin, alt_ref)

    reg = np.polyfit(np.log(model[ref[0]:ref[1]]),
                     np.log(signal[ref[0]:ref[1]]),
                     1)

    return np.exp(reg[0] * np.log(model) + reg[1])
