"""Testando a transformação de dados de binário para xarray"""

import os
import matplotlib.pyplot as plt
from lidarpy import GetData

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")]
files.sort()

data = GetData(directory, files)

lidar_data = data.get_xarray()

print(lidar_data)

lidar_data[:, 1, 0:3000].plot(x="time", y="altitude", figsize=(12, 7))
plt.show()

(lidar_data[:, 1, 0:3000] * lidar_data[:, 1, 0:3000].coords["altitude"] ** 2)\
    .plot(x="time", y="altitude", figsize=(12, 7))

plt.show()

data.to_netcdf("data/netcdf", "lidar_data")
