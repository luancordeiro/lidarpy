import numpy as np
import xarray as xr
import pandas as pd
from lidarpy.inversion.raman import Raman
from lidarpy.plot.plotter import compare_w_sol
from lidarpy.data.manipulation import groupby_nbins, signal_smoother
from scipy.signal import savgol_filter
from lidarpy.data.manipulation import remove_background, dead_time_correction
import matplotlib.pyplot as plt

df_elastic_signals = pd.read_csv("data/raman/elastic_signal.txt")

df_temp_pressure = pd.read_csv("data/netcdf/earlinet_pres_temp.txt", " ")

ds_solution = xr.open_dataset("data/netcdf/earlinet_solution.nc")

ds_data = xr.open_dataset("data/netcdf/earlinet_data.nc")

n_bins_mean = 1
n_bins_group = 4
alt_min = 300
alt_max = 20_000
ds_data = (ds_data
           # .pipe(dead_time_correction, 0)
           .mean("time")
           .pipe(remove_background, [28_000, 30_000])
           .pipe(groupby_nbins, n_bins_group)
           .rolling(rangebin=n_bins_mean, center=True)
           .mean()
           .dropna("rangebin")
           .sel(rangebin=slice(alt_min, alt_max)))

df_temp_pressure = (df_temp_pressure
                    .assign(Temperature=lambda x: x.Temperature + 273.15)
                    .assign(Pressure=lambda x: x.Pressure * 100)
                    .groupby(df_temp_pressure.index // n_bins_group)
                    .mean()
                    .rolling(n_bins_mean, center=True)
                    .mean()
                    .dropna())

df_temp_pressure = df_temp_pressure[(df_temp_pressure.Altitude >= alt_min) & (df_temp_pressure.Altitude <= alt_max)]

print(ds_data.phy.shape)
print(df_temp_pressure.shape)
print(df_temp_pressure.head())
print(ds_data.coords["rangebin"])
print(ds_solution)

wavelengths = [355, 387]

raman = Raman(ds_data,
              wavelengths[0],
              wavelengths[1],
              1.8,
              df_temp_pressure["Pressure"].to_numpy(),
              df_temp_pressure["Temperature"].to_numpy(),
              [10000, 12000])

alpha, beta, lr = raman.fit(diff_window=5)

max_range = 6000

indx_sol = (ds_solution.coords["rangebin"].data < max_range)

compare_w_sol(raman.rangebin[raman.rangebin < max_range],
              alpha[raman.rangebin < max_range],
              ds_solution.coords["rangebin"][indx_sol],
              ds_solution.sel(channel=f"{wavelengths[0]}_1").extinction.data[indx_sol],
              0)

compare_w_sol(raman.rangebin[raman.rangebin < max_range],
              beta[raman.rangebin < max_range],
              ds_solution.coords["rangebin"][indx_sol],
              ds_solution.sel(channel=f"{wavelengths[0]}_1").backscatter.data[indx_sol],
              1)

compare_w_sol(raman.rangebin[raman.rangebin < max_range],
              lr[raman.rangebin < max_range],
              ds_solution.coords["rangebin"][indx_sol],
              ds_solution.sel(channel=f"{wavelengths[0]}_1").lidar_ratio.data[indx_sol],
              2)

print(lr[raman.rangebin < max_range].mean())
