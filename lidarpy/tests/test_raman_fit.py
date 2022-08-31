from lidarpy import GetData
from lidarpy.data.manipulation import remove_background
from lidarpy.inversion.raman import Raman
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")]

data = GetData(directory, files)

lidar_data = data.get_xarray()

lidar_data = lidar_data.pipe(remove_background, [25_000, 80_000])

lidar_data = lidar_data.mean("time")

ds = lidar_data[[1, 3], 1100:2100].rolling(altitude=7).mean().dropna("altitude")

# plt.plot(lidar_data.coords["altitude"][:3000], lidar_data.sel(wavelength="355_1").data[:3000])
# plt.show()
#
# plt.plot(lidar_data.coords["altitude"][:3000],
#          lidar_data.sel(wavelength="355_1").data[:3000] * lidar_data.coords["altitude"][:3000] ** 2)
# plt.show()

df_sonde = pd.read_csv("data/sonde_data.txt")

# df_sonde.plot(x="alt", subplots=True, figsize=(12, 7))
# plt.show()

f_temp = interp1d(df_sonde["alt"].to_numpy(), df_sonde["temp"].to_numpy())
f_pres = interp1d(df_sonde["alt"].to_numpy(), df_sonde["pres"].to_numpy())

z = ds.coords["altitude"].data

temperature = f_temp(z)
pressure = f_pres(z)

print(temperature.shape, pressure.shape)

raman = Raman(ds,
              355,
              387,
              0,
              pressure,
              temperature,
              10_000)
raman.fit()
