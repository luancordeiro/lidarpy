import numpy as np
import xarray as xr
import pandas as pd
from lidarpy.inversion.raman import Raman
from lidarpy.plot.plotter import compare_w_sol
from lidarpy.data.manipulation import groupby_nbins

df_elastic_signals = pd.read_csv("data/raman/elastic_signal.txt")
df_raman_signals = pd.read_csv("data/raman/raman_signal.txt")
df_temp_pressure = pd.read_csv("data/raman/temp_pressure.txt")
df_sol = pd.read_csv("data/raman/sol.txt")

nbins = 1

df_sol = df_sol.assign(Backscatter=lambda x: x["Extinction"] / x["Lidarratio"])

print(df_sol.columns)
print(df_temp_pressure.head())

ds = xr.DataArray([df_elastic_signals["MeanSignal"], df_raman_signals["MeanSignal"]],
                  coords=(["355_1", "387_1"], df_elastic_signals["Altitude"].to_numpy()),
                  dims=["wavelength", "altitude"])
ds = ds.pipe(groupby_nbins, nbins)
df_temp_pressure = df_temp_pressure.groupby(df_temp_pressure.index // nbins).mean()

pressure = df_temp_pressure["Pressure"].to_numpy()
temperature = df_temp_pressure["Temperature"].to_numpy()

print(ds.shape)
print(pressure.shape)

indx = np.where((ds.coords["altitude"] > 300) & (ds.coords["altitude"] < 14_000))[0]
alpha, beta, lr = Raman(ds.isel(altitude=indx),
                        355,
                        387,
                        1.8,
                        pressure[indx],
                        temperature[indx],
                        12_000).fit()

compare_w_sol(ds.coords["altitude"].data[indx],
              alpha,
              df_sol["Extinction"][indx],
              "Extinction")

compare_w_sol(ds.coords["altitude"].data[indx],
              beta,
              df_sol["Backscatter"][indx],
              "Backscatter")
