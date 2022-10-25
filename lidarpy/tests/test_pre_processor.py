from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, get_uncertainty, dead_time_correction, z_finder
from lidarpy.data.pre_processor import pre_processor
from lidarpy.clouds.cloud_detection import CloudFinder
import os
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")]
ds = GetData(directory, files[:30]).get_xarray().isel(channel=[1, 3])

original_data = ds.copy()

original_data = (
    original_data
    .pipe(remove_background, [original_data.coords["rangebin"][-1] - 3000, original_data.coords["rangebin"][-1]])
    .pipe(dead_time_correction, 0.004)
    .mean("time")
    .pipe(get_uncertainty, 355, 600 * 30)
)


def process(lidar_data):
    lidar_data = (
        lidar_data
        .pipe(remove_background, [original_data.coords["rangebin"][-1] - 3000, original_data.coords["rangebin"][-1]])
        .pipe(dead_time_correction, 0.004)
        .mean("time")
    )

    return lidar_data


ds = ds.pipe(pre_processor, 50, process, True)

ind_min = int(5000 // 7.5)
ind_max = int(20_000 // 7.5)

plt.plot(ds.coords["rangebin"].data[ind_min:ind_max], ds.isel(channel=0).sigma.data[ind_min:ind_max], "-", label="MC")
plt.plot(original_data.coords["rangebin"].data[ind_min:ind_max], original_data.sigma.data[ind_min:ind_max], "--", label="diego")
plt.legend()
plt.grid()
plt.show()

plt.plot(ds.coords["rangebin"].data[ind_min:ind_max],
         (ds.isel(channel=0).sigma.data - original_data.sigma.data)[ind_min:ind_max] ** 2,
         "-")
plt.grid()
plt.show()

lidar_data = ds.sel(channel="355_1", rangebin=slice(7.5, 30_001))

cloud = CloudFinder(lidar_data, 355, 378, 5, 735036.004918982)
z_base, z_top, z_max_capa, nfz_base, nfz_top, nfz_max_capa = cloud.fit()

print("z_base", z_base)
print("z_top", z_top)

rcs = (lidar_data.phy * lidar_data.coords["rangebin"] ** 2)
indx_base = z_finder(lidar_data.coords["rangebin"].data, z_base)
indx_top = z_finder(lidar_data.coords["rangebin"].data, z_top)


plt.plot(lidar_data.coords["rangebin"], lidar_data.phy * lidar_data.coords["rangebin"] ** 2, "k-", alpha=0.6)
plt.plot([lidar_data.coords["rangebin"][indx_base]] * 2,
         [min(rcs), max(rcs)],
         "b--",
         label="base")
plt.plot([lidar_data.coords["rangebin"][indx_top]] * 2,
         [min(rcs), max(rcs)],
         "y--",
         label="top")

plt.ylabel("RCS")
plt.xlabel("Altitude (m)")
plt.legend()
plt.grid()
plt.show()
