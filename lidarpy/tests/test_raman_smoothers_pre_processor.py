import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from lidarpy.inversion.raman import Raman
from lidarpy.data.manipulation import groupby_nbins
from lidarpy.data.manipulation import remove_background, dead_time_correction, get_uncertainty
from lidarpy.data.raman_smoothers import get_savgol_filter, get_gaussian_filter, get_beta_savgol, get_beta_gaussian
from lidarpy.data.pre_processor import pre_processor

df_temp_pressure = pd.read_csv("data/netcdf/earlinet_pres_temp.txt", " ")
ds_solution = xr.open_dataset("data/netcdf/earlinet_solution.nc")
ds_data = xr.open_dataset("data/netcdf/earlinet_data.nc")

wavelengths = [355, 387]  # 355 e 387 ou 532 e 608
n_bins_group = 4
n_bins_mean = 5


def process(lidar_data):
    return (
        lidar_data
        .pipe(remove_background, [28_000, 30_000])
        .mean("time")
        .pipe(groupby_nbins, n_bins_group)
        .rolling(rangebin=n_bins_mean, center=True)
        .mean()
    )


alt_min = 300
alt_max = 20_000
ds_data = (ds_data.isel(channel=[0, 3])
           .pipe(pre_processor, 50, process, True)
           .sel(rangebin=slice(alt_min, alt_max)))

df_temp_pressure = (df_temp_pressure
                    .assign(Temperature=lambda x: x.Temperature + 273.15)
                    .assign(Pressure=lambda x: x.Pressure * 100)
                    .groupby(df_temp_pressure.index // n_bins_group)
                    .mean()
                    .rolling(n_bins_mean, center=True)
                    .mean())

df_temp_pressure = df_temp_pressure[(df_temp_pressure.Altitude >= alt_min) & (df_temp_pressure.Altitude <= alt_max)]

raman = Raman(ds_data.isel(rangebin=slice(n_bins_mean, 9999)),
              wavelengths[0],
              wavelengths[1],
              1.8,
              df_temp_pressure["Pressure"].to_numpy()[n_bins_mean:],
              df_temp_pressure["Temperature"].to_numpy()[n_bins_mean:],
              [10000, 12000])

smoothers = {
    # "SG2_W5": get_savgol_filter(5, 2),
    # "SG2_W7": get_savgol_filter(7, 2),
    # "SG2_W9": get_savgol_filter(9, 2),
    # "SG2_W11": get_savgol_filter(11, 2),
    # "SG2_W21": get_savgol_filter(21, 2),
    # "SG2_W23": get_savgol_filter(23, 2),
    "SG2_W31": get_savgol_filter(31, 2),
    # "SG3_W5": get_savgol_filter(5, 3),
    # "SG3_W7": get_savgol_filter(7, 3),
    # "SG3_W9": get_savgol_filter(9, 3),
    # "G0.5": get_gaussian_filter(0.5),
    # "G0.7": get_gaussian_filter(0.7),
    # "G0.9": get_gaussian_filter(0.9),
    # "G0.95": get_gaussian_filter(0.95),
    "G0.99": get_gaussian_filter(0.99),
}

beta_smoothers = {
    "SG2_W5": get_beta_savgol(5, 2),
    "SG2_W7": get_beta_savgol(7, 2),
    "SG2_W9": get_beta_savgol(9, 2),
    "SG2_W11": get_beta_savgol(11, 2),
    "SG2_W21": get_beta_savgol(21, 2),
    "SG2_W23": get_beta_savgol(23, 2),
    "SG2_W31": get_beta_savgol(31, 2),
    "SG3_W5": get_beta_savgol(5, 3),
    "SG3_W7": get_beta_savgol(7, 3),
    "SG3_W9": get_beta_savgol(9, 3),
    "G0.5": get_beta_gaussian(0.5),
    "G0.7": get_beta_gaussian(0.7),
    "G0.9": get_beta_gaussian(0.9),
    "G0.95": get_beta_gaussian(0.95),
    "G0.99": get_beta_gaussian(0.99),
}

max_range = 6000
indx_sol = (ds_solution.coords["rangebin"].data < max_range)

for window in [5, 7, 9, 11, 13, 15]:
    alphas = []
    betas = []
    lidar_ratios = []
    for key, value in smoothers.items():
        alpha_temp, beta_temp, lidar_ratio_temp = raman.fit(diff_strategy=value,
                                                            diff_window=window,
                                                            beta_smoother=beta_smoothers[key])

        alphas.append(alpha_temp)
        betas.append(beta_temp)
        lidar_ratios.append(lidar_ratio_temp)

    plt.plot(ds_solution.coords["rangebin"][indx_sol],
             ds_solution.sel(channel=f"{wavelengths[0]}_1").extinction.data[indx_sol],
             "k-",
             label="solution")
    for alpha, key in zip(alphas, smoothers.keys()):
        plt.plot(raman.rangebin[raman.rangebin < max_range], alpha[raman.rangebin < max_range], "--", label=key)
    plt.title(f"window={window}")
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(ds_solution.coords["rangebin"][indx_sol],
             ds_solution.sel(channel=f"{wavelengths[0]}_1").lidar_ratio.data[indx_sol],
             "k-",
             label="solution")
    for lidar_ratio, key in zip(lidar_ratios, smoothers.keys()):
        plt.plot(raman.rangebin[raman.rangebin < max_range], lidar_ratio[raman.rangebin < max_range], "--", label=key)
    plt.title(f"window={window}")
    plt.legend()
    plt.grid()
    plt.show()
