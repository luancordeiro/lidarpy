from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, atmospheric_interpolation, molecular_raman_model
import os
import matplotlib.pyplot as plt
import pandas as pd

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")]

lidar_data = (GetData(directory, files)
              .get_xarray()
              .pipe(remove_background, [120_000, 125_000])
              .mean("time"))

ds = lidar_data.isel(channel=3, rangebin=slice(500, 3000))

temperature, pressure = atmospheric_interpolation(ds.coords["rangebin"].data,
                                                  pd.read_csv("data/sonde_data.txt"))

model = molecular_raman_model(ds, 355, 387, pressure, temperature, [5000, 11_000])

z = ds.coords["rangebin"].data
plt.plot(z, ds.phy.data * z ** 2, "-", color="black", label="Signal")
plt.plot(z, model * z ** 2, "--", label="molecular model")

plt.grid()
plt.legend()
plt.xlabel("Altitude (m)")
plt.show()

