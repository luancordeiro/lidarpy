import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2, chisquare
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def dif(y: np.array, x: np.array, window: int, weights: np.array = None):
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


class DifChi2:
    def __init__(self, y: np.array, x: np.array, window: int, weights: np.array = None):
        if window % 2 == 0:
            raise ValueError("window must be odd.")

        self.ranged_corrected_signal = y
        self.rangebin = x
        self.window = window
        self.weights = weights

        win = self.window // 2
        dif_ranged_corrected_signal = []
        fit_order = []
        for i in range(win, len(y) - win - 10 - 1):
            dif, chosen_model = self._fit_chi2(i - win, i + win + 1)
            dif_ranged_corrected_signal.append(dif)
            fit_order.append(chosen_model + 1)

        plt.plot(fit_order, np.arange(len(fit_order)), "o")
        plt.grid()
        plt.show()

        for i in range(self.window // 2):
            dif_ranged_corrected_signal.insert(0, dif_ranged_corrected_signal[0])

        while len(dif_ranged_corrected_signal) != len(y):
            dif_ranged_corrected_signal += [dif_ranged_corrected_signal[-1]]

        self.fit_order = fit_order
        self.dif_ranged_corrected_signal = np.array(dif_ranged_corrected_signal)

    def _comp_chi2(self, o: np.array, e: np.array):
        return sum(((o - e) ** 2) / e)

    def _fit_chi2(self, init, final) -> tuple:
        y_fit = self.ranged_corrected_signal[init: final]
        # y_mean = y_fit.mean()
        # y_fit /= y_mean
        x_fit = self.rangebin[init: final]
        weight_fit = None if self.weights is None else self.weights[init: final]

        models = [np.poly1d(np.polyfit(x_fit, y_fit, n, w=weight_fit)) for n in range(1, self.window - 1)]

        chi2s = [self._comp_chi2(p(x_fit), y_fit) for p in models]

        residuals = [p(x_fit) - y_fit for p in models]

        cdfs = [chi2.cdf(comp_chi2, self.window - (2 + n)) for n, comp_chi2 in enumerate(chi2s)]

        chosen_model = (abs(np.array(cdfs) - 0.5)).argmin()

        # fig, axs = plt.subplots(2, sharex=True)
        # axs[0].plot(x_fit, y_fit, "k*", label="data")
        # for i, model in enumerate(models):
        #     axs[0].plot(np.linspace(min(x_fit), max(x_fit)),
        #                 model(np.linspace(min(x_fit), max(x_fit))),
        #                 "--",
        #                 label=f"polyorder={1 + i} \n chi2={chi2s[i]} \n cdfs={cdfs[i]} \n")
        #     axs[1].plot(x_fit, residuals[i], "o", label=f"polyorder={1 + i}")
        # plt.title(f"polyorder selected model = {chosen_model + 1}")
        # axs[0].legend()
        # axs[1].legend()
        # plt.grid()
        # plt.show()

        model = models[chosen_model]

        return (
            (model(x_fit[len(x_fit) // 2] + 1e-6) - model(x_fit[len(x_fit) // 2])) / 1e-6,
            chosen_model
        )

    def get_dif_ranged_corrected_signal(self) -> np.array:
        return self.dif_ranged_corrected_signal

    def apply(self, num_density) -> np.array:
        win = self.window // 2
        dif_num_density = []
        for n, i in enumerate(range(win, len(num_density) - win - 10 - 1)):
            init, final = i - win, i + win + 1
            y_fit = num_density[init: final]
            x_fit = self.rangebin[init: final]

            model = np.poly1d(np.polyfit(y_fit, x_fit, self.fit_order[n]))

            dif_ = (model(x_fit[win] + 0.000001) - model(x_fit[win])) / 0.000001
            dif_num_density.append(dif_)

        for i in range(self.window // 2):
            dif_num_density.insert(0, dif_num_density[0])

        while len(dif_num_density) != len(num_density):
            dif_num_density += [dif_num_density[-1]]

        return np.array(dif_num_density)


def diff_chi2_test(num_density: np.array, ranged_corrected_signal: np.array, rangebin: np.array,
                   diff_window: int, inelastic_uncertainty=None):
    weights = None if inelastic_uncertainty is None else 1 / (inelastic_uncertainty * rangebin ** 2) ** 2
    diff = DifChi2(ranged_corrected_signal, rangebin, diff_window, weights)
    dif_ranged_corrected_signal = diff.get_dif_ranged_corrected_signal()
    dif_num_density = diff.apply(num_density)
    return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)


def diff_linear_regression(num_density: np.array, ranged_corrected_signal: np.array, rangebin: np.array,
                           diff_window: int, inelastic_uncertainty=None):

    weights = None if inelastic_uncertainty is None else 1 / (inelastic_uncertainty * rangebin ** 2) ** 2

    dif_ranged_corrected_signal = dif(ranged_corrected_signal, rangebin, diff_window, weights)
    dif_num_density = dif(num_density, rangebin, diff_window)
    return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)


def diff_without_smooth(num_density: np.array, ranged_corrected_signal: np.array, rangebin: np.array,
                        diff_window: int, inelastic_uncertainty=None):

    weights = None if inelastic_uncertainty is None else 1 / (inelastic_uncertainty * rangebin ** 2) ** 2

    dif_ranged_corrected_signal = np.gradient(ranged_corrected_signal) / np.gradient(rangebin)
    dif_num_density = np.gradient(num_density) / np.gradient(rangebin)
    return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)


def get_savgol_filter2(window_length, polyorder):
    def diff_savgol_filter(num_density: np.array, ranged_corrected_signal: np.array, rangebin: np.array,
                           diff_window: int, inelastic_uncertainty=None):

        num_density = savgol_filter(num_density, window_length, polyorder)
        ranged_corrected_signal = savgol_filter(ranged_corrected_signal, window_length, polyorder)

        weights = None if inelastic_uncertainty is None else 1 / (inelastic_uncertainty * rangebin ** 2) ** 2

        dif_ranged_corrected_signal = np.gradient(ranged_corrected_signal) / np.gradient(rangebin)
        dif_num_density = np.gradient(num_density) / np.gradient(rangebin)
        return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)

    return diff_savgol_filter


def get_savgol_filter3(window_length, polyorder):
    def diff_savgol_filter(num_density: np.array, ranged_corrected_signal: np.array, rangebin: np.array,
                           diff_window: int, inelastic_uncertainty=None):

        weights = None if inelastic_uncertainty is None else 1 / (inelastic_uncertainty * rangebin ** 2) ** 2

        delta = rangebin[1] - rangebin[0]
        dif_num_density = savgol_filter(num_density, window_length, polyorder, 1, delta)
        dif_ranged_corrected_signal = savgol_filter(ranged_corrected_signal, window_length, polyorder, 1, delta)

        return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)

    return diff_savgol_filter


def get_savgol_filter(window_length, polyorder):
    def diff_savgol_filter(num_density: np.array, ranged_corrected_signal: np.array, rangebin: np.array,
                           diff_window: int, inelastic_uncertainty=None):

        num_density = savgol_filter(num_density, window_length, polyorder)
        ranged_corrected_signal = savgol_filter(ranged_corrected_signal, window_length, polyorder)

        weights = None if inelastic_uncertainty is None else 1 / (inelastic_uncertainty * rangebin ** 2) ** 2

        dif_num_density = dif(num_density, rangebin, diff_window)
        dif_ranged_corrected_signal = dif(ranged_corrected_signal, rangebin, diff_window, weights)
        return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)

    return diff_savgol_filter


def get_gaussian_filter(sigma):
    def diff_savgol_filter(num_density: np.array, ranged_corrected_signal: np.array, rangebin: np.array,
                           diff_window: int, inelastic_uncertainty=None):
        ranged_corrected_signal = gaussian_filter(ranged_corrected_signal, sigma)
        num_density = gaussian_filter(num_density, sigma)

        weights = None if inelastic_uncertainty is None else 1 / (inelastic_uncertainty * rangebin ** 2) ** 2

        dif_num_density = dif(num_density, rangebin, diff_window)
        dif_ranged_corrected_signal = dif(ranged_corrected_signal, rangebin, diff_window, weights)
        return (dif_num_density / num_density) - (dif_ranged_corrected_signal / ranged_corrected_signal)

    return diff_savgol_filter


def get_beta_gaussian(sigma):
    def smoother_(x):
        return gaussian_filter(x, sigma)

    return smoother_


def get_beta_savgol(window_length, polyorder):
    def smoother_(x):
        return savgol_filter(x, window_length, polyorder)

    return smoother_
