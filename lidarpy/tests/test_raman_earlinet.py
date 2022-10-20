import numpy as np
import xarray as xr
import pandas as pd
from lidarpy.inversion.raman import Raman
from lidarpy.plot.plotter import compare_w_sol
from lidarpy.data.manipulation import groupby_nbins, signal_smoother
from scipy.signal import savgol_filter
from lidarpy.data.manipulation import remove_background
import matplotlib.pyplot as plt

df_elastic_signals = pd.read_csv("data/raman/elastic_signal.txt")
df_raman_signals = pd.read_csv("data/raman/raman_signal.txt")
df_temp_pressure = pd.read_csv("data/raman/temp_pressure.txt")
df_sol = pd.read_csv("data/raman/sol.txt")
df_sol = df_sol.assign(Backscatter=lambda x: x["Extinction"] / x["Lidarratio"])

# window = 5
# signals = [signal_smoother(df_elastic_signals["MeanSignal"], df_elastic_signals["Altitude"].to_numpy(), window),
#            signal_smoother(df_raman_signals["MeanSignal"], df_elastic_signals["Altitude"].to_numpy(), window)]

signals = [df_elastic_signals["MeanSignal"], df_raman_signals["MeanSignal"]]


ds = xr.Dataset(
    {
        "phy": xr.DataArray(signals,
                            coords=(["355_1", "387_1"], df_elastic_signals["Altitude"].to_numpy()),
                            dims=["channel", "rangebin"])
    }
)

plt.plot(ds.coords["rangebin"].data,
         ds.sel(channel="355_1").phy.data * ds.coords["rangebin"].data ** 2)

nbins = 4
window = 3
ds = (ds
      .pipe(remove_background, [28_000, 30_000])
      .pipe(groupby_nbins, nbins)
      .rolling(rangebin=window, center=True)
      .mean()
      .dropna("rangebin")
      )

df_temp_pressure = df_temp_pressure.groupby(df_temp_pressure.index // nbins).mean()

plt.plot(ds.coords["rangebin"].data,
         ds.sel(channel="355_1").phy.data * ds.coords["rangebin"].data ** 2)
plt.xlabel("Altitude (m)")
plt.ylabel("RCS")
plt.grid()
plt.show()

indx = np.where((ds.coords["rangebin"] > 300) & (ds.coords["rangebin"] < 20000))[0]
raman = Raman(ds.isel(rangebin=indx),
              355,
              387,
              1.8,
              df_temp_pressure["Pressure"].to_numpy()[indx],
              df_temp_pressure["Temperature"].to_numpy()[indx],
              [10000, 12000])

alpha, beta, lr = raman.fit(diff_window=5)

max_range = 7000

indx_sol = (df_sol["Altitude"] > 300) & (df_sol["Altitude"] < max_range)

compare_w_sol(raman.rangebin[raman.rangebin < max_range],
              alpha[raman.rangebin < max_range],
              df_sol["Altitude"][indx_sol],
              df_sol["Extinction"][indx_sol],
              0)

compare_w_sol(raman.rangebin[raman.rangebin < max_range],
              beta[raman.rangebin < max_range],
              df_sol["Altitude"][indx_sol],
              df_sol["Backscatter"][indx_sol],
              1)

compare_w_sol(raman.rangebin[raman.rangebin < max_range],
              lr[raman.rangebin < max_range],
              df_sol["Altitude"][indx_sol],
              df_sol["Lidarratio"][indx_sol],
              2)

print(lr[raman.rangebin < max_range].mean())
