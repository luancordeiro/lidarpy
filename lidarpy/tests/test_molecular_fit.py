from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, atmospheric_interpolation, molecular_model
from lidarpy.inversion.klett import Klett
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")]

data = GetData(directory, files)

lidar_data = data.get_xarray()

lidar_data = lidar_data.pipe(remove_background, [25_000, 80_000])

lidar_data = lidar_data.mean("time")

ds = lidar_data.isel(wavelength=1, altitude=slice(500, 3000))

temperature, pressure = atmospheric_interpolation(ds.coords["altitude"].data,
                                                  pd.read_csv("data/sonde_data.txt"))

model = molecular_model(ds, 355, pressure, temperature, [5000, 11_000])

z = ds.coords["altitude"].data
plt.plot(z, ds.phy.data, "-", color="black", label="Signal")
plt.plot(z, model, "--", label="molecular model")

plt.grid()
plt.legend()
plt.xlabel("Altitude (m)")
plt.show()

plt.plot(z, ds.phy.data - model)
plt.show()

ref = lidar_data.coords["altitude"].sel(altitude=[5000, 11_000], method="nearest").data
ref = np.where((z == ref[0]) | (z == ref[1]))[0]

plt.plot(z[ref[0]:ref[1]], (ds.phy.data - model)[ref[0]:ref[1]])
plt.show()
