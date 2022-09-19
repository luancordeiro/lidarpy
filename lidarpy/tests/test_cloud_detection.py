from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, atmospheric_interpolation
from lidarpy.inversion.transmittance import Transmittance
import os
import matplotlib.pyplot as plt
import pandas as pd

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")]

data = GetData(directory, files)

lidar_data = data.get_xarray()

lidar_data = lidar_data.pipe(remove_background, [25_000, 80_000])

lidar_data = lidar_data.mean("time")

print(lidar_data.sel(wavelength="355_1").data[:50].round(3))

plt.plot(lidar_data.coords["altitude"], lidar_data.sel(wavelength="355_1") * lidar_data.coords["altitude"] ** 2, "--")
plt.ylabel("RCS")
plt.xlabel("Altitude (m)")
plt.grid()
plt.show()
