import numpy as np
import xarray as xr
import pandas as pd
from lidarpy.inversion.raman import Raman, diff_
from lidarpy.plot.plotter import compare_w_sol

df_elastic_signals = pd.read_csv("data/raman/elastic_signal.txt")
df_raman_signals = pd.read_csv("data/raman/raman_signal.txt")
df_temp_pressure = pd.read_csv("data/raman/temp_pressure.txt")
df_sol = pd.read_csv("data/raman/sol.txt")

print(df_sol.c)

indx = np.arange(25, 1000)

pressure = df_temp_pressure["Pressure"].to_numpy() * 100
temperature = df_temp_pressure["Temperature"].to_numpy() + 273.15

ds = xr.DataArray([df_elastic_signals["MeanSignal"], df_raman_signals["MeanSignal"]],
                  coords=(["355_1", "387_1"], df_elastic_signals["Altitude"].to_numpy()),
                  dims=["wavelength", "altitude"])

print(ds.shape)

alpha, beta, lr = Raman(ds.isel(altitude=indx),
                        355,
                        387,
                        1.8,
                        pressure[indx],
                        temperature[indx],
                        12_000).fit(diff_)

compare_w_sol(ds.coords["altitude"],
              df_sol[""])