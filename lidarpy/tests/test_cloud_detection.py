from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from lidarpy.clouds.cloud_detection import CloudFinder

df = pd.read_csv("data/cloud_detection_tester.txt")

lidar_data = xr.DataArray(df["2"], dims=["altitude"], coords=[df["1"]])

sigma = df["3"].to_numpy()
jdz = 735036.004918982

lidar_data = lidar_data.pipe(remove_background, [25_000, 80_000])

# plt.plot(lidar_data.coords["altitude"], lidar_data * lidar_data.coords["altitude"] ** 2, "--")
# plt.ylabel("RCS")
# plt.xlabel("Altitude (m)")
# plt.grid()
# plt.show()
# #
# plt.plot(lidar_data.coords["altitude"], sigma * lidar_data.coords["altitude"] ** 2, "--")
# plt.ylabel("sigma")
# plt.xlabel("Altitude (m)")
# plt.grid()
# plt.show()

cloud = CloudFinder(lidar_data, sigma, 355, 378, 5, jdz)
z_base, z_top, z_max_capa, nfz_base, nfz_top, nfz_max_capa = cloud.fit()

print(z_base)
indx_base = lidar_data.coords["altitude"].sel(altitude=z_base, method="nearest").data
indx_base = lidar_data.coords["altitude"].isin(indx_base)
indx_top = lidar_data.coords["altitude"].sel(altitude=z_top, method="nearest").data
indx_top = lidar_data.coords["altitude"].isin(indx_top)

plt.plot(lidar_data.coords["altitude"], lidar_data * lidar_data.coords["altitude"] ** 2, "r--")
plt.plot(lidar_data.coords["altitude"][indx_base],
         (lidar_data * lidar_data.coords["altitude"] ** 2)[indx_base],
         "b*")
plt.plot(lidar_data.coords["altitude"][indx_top],
         (lidar_data * lidar_data.coords["altitude"] ** 2)[indx_top],
         "y*")
plt.ylabel("RCS")
plt.xlabel("Altitude (m)")
plt.grid()
plt.show()
