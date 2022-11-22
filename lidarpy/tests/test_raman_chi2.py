import numpy as np
import xarray as xr
import pandas as pd
from lidarpy.inversion.raman import Raman
from lidarpy.plot.plotter import compare_w_sol
from lidarpy.data.manipulation import groupby_nbins
from lidarpy.data.manipulation import remove_background, dead_time_correction
from lidarpy.data.raman_smoothers import diff_chi2_test, diff_linear_regression

df_temp_pressure = pd.read_csv("data/netcdf/earlinet_pres_temp.txt", " ")
ds_solution = xr.open_dataset("data/netcdf/earlinet_solution.nc")
ds_data = xr.open_dataset("data/netcdf/earlinet_data.nc")

wavelengths = [355, 387]  # 355 e 387 ou 532 e 608

n_bins_mean = 3
n_bins_group = 5
alt_min = 0
alt_max = 20_000
ds_data = (ds_data
           .pipe(dead_time_correction, 0)
           .pipe(remove_background, [28_000, 30_000])
           .mean("time")
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

raman = Raman(ds_data.isel(rangebin=slice(n_bins_mean, 9999)),
              wavelengths[0],
              wavelengths[1],
              1.8,
              df_temp_pressure["Pressure"].to_numpy()[n_bins_mean:],
              df_temp_pressure["Temperature"].to_numpy()[n_bins_mean:],
              [10000, 12000])

alpha, beta, lr = raman.fit(
    diff_window=5,
    diff_strategy=diff_chi2_test
)

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
