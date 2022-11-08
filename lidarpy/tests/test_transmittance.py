import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lidarpy.inversion.transmittance import GetCod
from lidarpy.data.manipulation import remove_background, remove_background_fit
from lidarpy.data.pre_processor import pre_processor

weak_cloud = 1

link = ["http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500.txt",
        "http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500_v2.txt"]

if weak_cloud:
    df_sol = pd.read_csv("data/sol_lalinet_weak_cloud.txt", delimiter="\t")

my_data = np.genfromtxt(link[weak_cloud])

ds = xr.DataArray(my_data[:, 1], dims=["rangebin"])
ds.coords["rangebin"] = my_data[:, 0]
print(ds.shape)
ds = xr.Dataset({"phy": ds})

df_sonde = pd.read_csv("data/sonde_lalinet.txt", delimiter="\t")

df_sonde = (df_sonde
            .assign(pressure=lambda x: x.pressure * 100)
            .assign(temperature=lambda x: x.temperature + 273.15)
            [["altitude", "pressure", "temperature"]])

print()
print(df_sonde.head())
print()

plt.plot(ds.phy.data * ds.coords["rangebin"] ** 2, label="antes")


def process(lidar_data):
    return (lidar_data
            .pipe(remove_background, [14_000, 15_000])
            .pipe(remove_background_fit,
                  355,
                  df_sonde.pressure.to_numpy(),
                  df_sonde.temperature.to_numpy(),
                  [14_000, 15_000]))


ds = ds.pipe(pre_processor, 500, process, True)

plt.plot(ds.phy.data * ds.coords["rangebin"] ** 2, label="depois")
plt.legend()
plt.show()

tau = GetCod(ds,
             [5800, 6150],
             355,
             df_sonde["pressure"].to_numpy(),
             df_sonde["temperature"].to_numpy(),
             mc_iter=300).fit()

print(tau)
