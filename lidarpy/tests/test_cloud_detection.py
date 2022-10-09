from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, get_uncertainty, dead_time_correction
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from lidarpy.clouds.cloud_detection import CloudFinder

df = pd.read_csv("data/cloud_detection_tester.txt")
lidar_data = xr.DataArray(df["2"], dims=["altitude"], coords=[df["1"]])
sigma = df["3"].to_numpy()

# directory = "data/binary"
# files = [file for file in os.listdir(directory) if file.startswith("RM")]
# data = GetData(directory, files)
#
# lidar_data = (
#     data
#     .get_xarray()
#     # .pipe(remove_background, [25_000, 80_000])
#     # .pipe(dead_time_correction, 0.004)
# )
#
# print(lidar_data.shape)
# # sigma = get_uncertainty(lidar_data,
# #                         355,
# #                         [25_000, 50_000],
# #                         9000)
# sigma = lidar_data.sel(wavelength="355_1", altitude=np.arange(7.5, 30_000, 7.5)).std("time", ddof=1)
# lidar_data = lidar_data.sel(wavelength="355_1", altitude=np.arange(7.5, 30_000, 7.5)).mean("time")
# sigma = sigma[:len(lidar_data)]

jdz = 735036.004918982

plt.plot(lidar_data.coords["altitude"], lidar_data * lidar_data.coords["altitude"] ** 2, "--")
plt.ylabel("RCS")
plt.xlabel("Altitude (m)")
plt.grid()
plt.show()
#
plt.plot(lidar_data.coords["altitude"], sigma * lidar_data.coords["altitude"] ** 2, "--")
plt.ylabel("sigma")
plt.xlabel("Altitude (m)")
plt.grid()
plt.show()

cloud = CloudFinder(lidar_data, sigma, 355, 378, 5, jdz)
z_base, z_top, z_max_capa, nfz_base, nfz_top, nfz_max_capa = cloud.fit()

rcs = (lidar_data * lidar_data.coords["altitude"] ** 2)
indx_base = lidar_data.coords["altitude"].sel(altitude=z_base, method="nearest").data
indx_base = lidar_data.coords["altitude"].isin(indx_base)
indx_top = lidar_data.coords["altitude"].sel(altitude=z_top, method="nearest").data
indx_top = lidar_data.coords["altitude"].isin(indx_top)

plt.plot(lidar_data.coords["altitude"], lidar_data * lidar_data.coords["altitude"] ** 2, "k-", alpha=0.6)
plt.plot([lidar_data.coords["altitude"][indx_base]] * 2,
         [min(rcs), max(rcs)],
         "b--",
         label="base")
plt.plot([lidar_data.coords["altitude"][indx_top]] * 2,
         [min(rcs), max(rcs)],
         "y--",
         label="top")

plt.ylabel("RCS")
plt.xlabel("Altitude (m)")
plt.legend()
plt.grid()
plt.show()
