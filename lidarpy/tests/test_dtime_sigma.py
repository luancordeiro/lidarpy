from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, get_uncertainty, dead_time_correction
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

df = pd.read_csv("data/cloud_detection_tester.txt")
ds_diego = xr.Dataset({"phy": xr.DataArray(df["2"], dims=["altitude"], coords=[df["1"]])})
sigma_diego = df["3"].to_numpy()

lidar_data = GetData("data/binary", [file for file in os.listdir("data/binary")
                                     if file.startswith("RM")][:25]).get_xarray()

lidar_copy = lidar_data.mean("time").phy.copy()

lidar_data = (lidar_data
              .mean("time")
              .pipe(remove_background, [119100, 122850])
              .pipe(dead_time_correction, 0.004))

sigma = lidar_data.pipe(get_uncertainty, 355, 600 * 15)

z = lidar_data.coords["altitude"].data
plt.plot(z[:3999], (sigma * z ** 2)[:3999])
plt.show()

plt.plot(z[:3999], sigma_diego * z[:3999] ** 2)
plt.show()

print("test")
