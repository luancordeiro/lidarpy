import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression


def dif(x: np.array, y: np.array, window: int, weights: np.array = None):
    def fit(init, final) -> None:
        y_fit = y[init: final].reshape(-1, 1)
        x_fit = x[init: final].reshape(-1, 1)
        weight_fit = None if weights is None else weights[init: final]

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


def diff_linear_regression(diff_window: int):
    def diff(raman):
        weights = (None if raman.inelastic_uncertainty is None
                   else 1 / (raman.inelastic_uncertainty * raman.rangebin ** 2) ** 2)

        ranged_corrected_signal = raman.inelastic_signal * raman.rangebin ** 2

        dif_ranged_corrected_signal = dif(raman.rangebin, ranged_corrected_signal, diff_window, weights)
        dif_num_density = dif(raman.rangebin, raman.raman_scatterer_numerical_density, diff_window)
        return ((dif_num_density / raman.raman_scatterer_numerical_density)
                - (dif_ranged_corrected_signal / ranged_corrected_signal))

    return diff


def get_savgol_filter(window_length, polyorder):
    def diff(raman):
        ranged_corrected_signal = raman.inelastic_signal * raman.rangebin ** 2
        num_density = savgol_filter(raman.raman_scatterer_numerical_density, window_length, polyorder)
        ranged_corrected_signal = savgol_filter(ranged_corrected_signal, window_length, polyorder)

        dif_ranged_corrected_signal = np.gradient(ranged_corrected_signal) / np.gradient(raman.rangebin)
        dif_num_density = np.gradient(num_density) / np.gradient(raman.rangebin)
        return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)

    return diff


def beta_smooth(raman):
    return raman.get_beta()["elastic_aer"]
