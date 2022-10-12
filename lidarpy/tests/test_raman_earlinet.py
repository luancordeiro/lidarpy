import numpy as np
import xarray as xr
import pandas as pd
from lidarpy.inversion.raman import Raman
from lidarpy.plot.plotter import compare_w_sol
from lidarpy.data.manipulation import groupby_nbins, signal_smoother
import matplotlib.pyplot as plt

df_elastic_signals = pd.read_csv("data/raman/elastic_signal.txt")
df_raman_signals = pd.read_csv("data/raman/raman_signal.txt")
df_temp_pressure = pd.read_csv("data/raman/temp_pressure.txt")
df_sol = pd.read_csv("data/raman/sol.txt")
df_sol = df_sol.assign(Backscatter=lambda x: x["Extinction"] / x["Lidarratio"])

window = 5

signals = [signal_smoother(df_elastic_signals["MeanSignal"], df_elastic_signals["Altitude"].to_numpy(), window),
           signal_smoother(df_raman_signals["MeanSignal"], df_elastic_signals["Altitude"].to_numpy(), window)]

ds = xr.Dataset(
    {
        "phy": xr.DataArray(signals,
                            coords=(["355_1", "387_1"], df_elastic_signals["Altitude"].to_numpy()),
                            dims=["wavelength", "altitude"])
    }
)

plt.plot(ds.coords["altitude"].data,
         ds.sel(wavelength="355_1").phy.data * ds.coords["altitude"].data ** 2)

nbins = 1
ds = ds.pipe(groupby_nbins, nbins)
df_temp_pressure = df_temp_pressure.groupby(df_temp_pressure.index // nbins).mean()

plt.plot(ds.coords["altitude"].data,
         ds.sel(wavelength="355_1").phy.data * ds.coords["altitude"].data ** 2)
plt.xlabel("Altitude (m)")
plt.ylabel("RCS")
plt.grid()
plt.show()

indx = np.where((ds.coords["altitude"] > 300) & (ds.coords["altitude"] < 9000))[0]
alpha, beta, lr = Raman(ds.isel(altitude=indx),
                        355,
                        387,
                        1.8,
                        df_temp_pressure["Pressure"].to_numpy()[indx],
                        df_temp_pressure["Temperature"].to_numpy()[indx],
                        12_000).fit(diff_window=15)

indx_sol = (df_sol["Altitude"] > 300) & (df_sol["Altitude"] < 9000)

compare_w_sol(ds.coords["altitude"].data[indx],
              alpha,
              df_sol["Altitude"][indx_sol],
              df_sol["Extinction"][indx_sol],
              0)

compare_w_sol(ds.coords["altitude"].data[indx],
              beta,
              df_sol["Altitude"][indx_sol],
              df_sol["Backscatter"][indx_sol],
              1)
