from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, get_uncertainty, dead_time_correction
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
    .pipe(remove_background, [75_000, 80_000])
    .pipe(dead_time_correction, 0.004)
    .mean("time")
    .pipe(get_uncertainty, 355, 600 * 30)
)


def process(lidar_data):
    lidar_data = (
        lidar_data
        .pipe(remove_background, [75_000, 80_000])
        .pipe(dead_time_correction, 0.004)
        .mean("time")
    )

    return lidar_data


ds = ds.pipe(pre_processor, 30, process, True)

ind_min = int(5000 // 7.5)
ind_max = int(20_000 // 7.5)

plt.plot(ds.coords["rangebin"].data[ind_min:ind_max], ds.isel(channel=0).uncertainty.data[ind_min:ind_max], "-", label="MC")
plt.plot(original_data.coords["rangebin"].data[ind_min:ind_max], original_data.sigma.data[ind_min:ind_max], "--", label="diego")
plt.legend()
plt.grid()
plt.show()

plt.plot(ds.coords["rangebin"].data[ind_min:ind_max],
         (ds.isel(channel=0).uncertainty.data - original_data.sigma.data)[ind_min:ind_max] ** 2,
         "-")
plt.grid()
plt.show()
