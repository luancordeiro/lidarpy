import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from lidarpy.inversion.transmittance import get_cod
from lidarpy.data.manipulation import remove_background, remove_background_fit
from lidarpy.inversion.klett import Klett
from lidarpy.plot.plotter import plot_3graph_std, compare_w_sol

weak_cloud = 0

link = ["http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500.txt",
        "http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500_v2.txt"]

if weak_cloud:
    df_sol = pd.read_csv("data/sol_lalinet_weak_cloud.txt", delimiter="\t")

my_data = np.genfromtxt(link[weak_cloud])

ds = xr.DataArray(my_data[:, 1], dims=["altitude"])
ds.coords["altitude"] = my_data[:, 0]
print(ds.shape)

df_sonde = pd.read_csv("data/sonde_lalinet.txt", delimiter="\t")

df_sonde = (df_sonde
            .assign(pressure=lambda x: x.pressure * 100)
            .assign(temperature=lambda x: x.temperature + 273.15)
            [["altitude", "pressure", "temperature"]])

print()
print(df_sonde.head())
print()

plt.plot(ds.data * ds.coords["altitude"] ** 2, label="antes")

ds = ds.pipe(remove_background, [10_000, 14_000])

ds = ds.pipe(remove_background_fit,
             355,
             df_sonde.pressure.to_numpy(),
             df_sonde.temperature.to_numpy(),
             [10_000, 14_000])

plt.plot(ds.data * ds.coords["altitude"] ** 2, label="depois")
plt.legend()
plt.show()

tau = get_cod(ds,
              [5800, 6150],
              355,
              df_sonde["pressure"].to_numpy(),
              df_sonde["temperature"].to_numpy())

print(tau)