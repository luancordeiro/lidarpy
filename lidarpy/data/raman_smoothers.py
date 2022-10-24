import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter


def diff(y: np.array, x: np.array, window: int, weights: np.array = None):
    def fit(init, final):
        y_fit = y[init: final].reshape(-1, 1)
        x_fit = x[init: final].reshape(-1, 1)

        if weights is None:
            linear_regession = LinearRegression().fit(x_fit, y_fit)
        else:
            weight_fit = weights[init: final]
            linear_regession = LinearRegression().fit(x_fit, y_fit, sample_weight=weight_fit)

        return linear_regession.coef_[0][0]

    if window % 2 == 0:
        raise ValueError("window must be odd.")

    win = window // 2
    diff_y = []
    for i in range(win, len(y) - win - 10 - 1):
        diff_y.append(fit(i - win, i + win + 1))

    for i in range(window // 2):
        diff_y.insert(0, diff_y[0])

    while len(diff_y) != len(y):
        diff_y += [diff_y[-1]]

    return np.array(diff_y)


def diff_linear_regression(num_density: np.array, ranged_corrected_signal: np.array, rangebin: np.array,
                           diff_window: int, inelastic_uncertainty=None):

    dif_num_density = diff(num_density, rangebin, diff_window, inelastic_uncertainty)
    dif_ranged_corrected_signal = diff(ranged_corrected_signal, rangebin, diff_window, inelastic_uncertainty)
    return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)


def get_savgol_filter(window_length, polyorder):
    def diff_savgol_filter(num_density: np.array, ranged_corrected_signal: np.array, rangebin: np.array,
                           diff_window: int, inelastic_uncertainty=None):
        ranged_corrected_signal = savgol_filter(ranged_corrected_signal, window_length, polyorder)
        num_density = savgol_filter(num_density, window_length, polyorder)

        dif_num_density = diff(num_density, rangebin, diff_window, inelastic_uncertainty)
        dif_ranged_corrected_signal = diff(ranged_corrected_signal, rangebin, diff_window, inelastic_uncertainty)
        return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)

    return diff_savgol_filter


def get_gaussian_filter(sigma):
    def diff_savgol_filter(num_density: np.array, ranged_corrected_signal: np.array, rangebin: np.array,
                           diff_window: int, inelastic_uncertainty=None):
        ranged_corrected_signal = gaussian_filter(ranged_corrected_signal, sigma)
        num_density = gaussian_filter(num_density, sigma)

        dif_num_density = diff(num_density, rangebin, diff_window, inelastic_uncertainty)
        dif_ranged_corrected_signal = diff(ranged_corrected_signal, rangebin, diff_window, inelastic_uncertainty)
        return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)

    return diff_savgol_filter
