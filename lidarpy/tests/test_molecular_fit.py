from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, atmospheric_interpolation, molecular_model
from lidarpy.inversion.klett import Klett
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")]

lidar_data = (GetData(directory, files)
              .get_xarray()
              .pipe(remove_background, [75_000, 80_000])
              .mean("time"))

ds = lidar_data.isel(channel=1, rangebin=slice(500, 3000))

temperature, pressure = atmospheric_interpolation(ds.coords["rangebin"].data,
                                                  pd.read_csv("data/sonde_data.txt"))

model = molecular_model(ds, 355, pressure, temperature, [5000, 11_000])

z = ds.coords["rangebin"].data
plt.plot(z, ds.phy.data * z ** 2, "-", color="black", label="Signal")
plt.plot(z, model * z ** 2, "--", label="molecular model")

plt.grid()
plt.legend()
plt.xlabel("Altitude (m)")
plt.show()

