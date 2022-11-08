from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, get_uncertainty, dead_time_correction, z_finder
from lidarpy.data.pre_processor import pre_processor
from lidarpy.clouds.cloud_detection import CloudFinder
import os
import matplotlib.pyplot as plt

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")][:80]
lidar_data = GetData(directory, files).get_xarray().isel(channel=1)

original_data = lidar_data.copy()

original_data = (
    original_data
    .pipe(remove_background, [original_data.coords["rangebin"][-1] - 3000, original_data.coords["rangebin"][-1]])
    .pipe(dead_time_correction, 0.004)
    .mean("time")
    .pipe(get_uncertainty, 355, 600 * len(files))
    .sel(rangebin=slice(1000, 30_001))
)


def process(lidar_data_):
    return (
        lidar_data_
        .pipe(remove_background, [original_data.coords["rangebin"][-1] - 3000, original_data.coords["rangebin"][-1]])
        .pipe(dead_time_correction, 0.004)
        .mean("time")
    )


lidar_data = lidar_data.pipe(pre_processor, 500, process, True).sel(rangebin=slice(1000, 30_001))

plt.plot(lidar_data.coords["rangebin"].data, lidar_data.sigma.data, "-",
         label="MC")
plt.plot(original_data.coords["rangebin"].data, original_data.sigma.data, "--",
         label="diego")
plt.legend()
plt.grid()
plt.show()

plt.plot(lidar_data.coords["rangebin"].data,
         (lidar_data.sigma.data - original_data.sigma.data) ** 2,
         "-")
plt.grid()
plt.show()

cloud = CloudFinder(lidar_data,
                    355,
                    3000,
                    5,
                    735036.004918982)

cloud_diego = CloudFinder(original_data,
                          355,
                          3000,
                          5,
                          735036.004918982)

z_base, z_top, *_ = cloud.fit()
z_base_diego, z_top_diego, *_ = cloud_diego.fit()

print("z_base", z_base)
print("z_top", z_top)
print("z_base_diego", z_base_diego)
print("z_top_diego", z_top_diego)

rcs = (lidar_data.phy * lidar_data.coords["rangebin"] ** 2)
indx_base = z_finder(lidar_data.coords["rangebin"].data, z_base)
indx_top = z_finder(lidar_data.coords["rangebin"].data, z_top)

indx_base_diego = z_finder(lidar_data.coords["rangebin"].data, z_base_diego)
indx_top_diego = z_finder(lidar_data.coords["rangebin"].data, z_top_diego)

min_, max_ = min(rcs), max(rcs)

plt.plot(lidar_data.coords["rangebin"], lidar_data.phy * lidar_data.coords["rangebin"] ** 2, "k-", alpha=0.6)
plt.plot([lidar_data.coords["rangebin"][indx_base]] * 2,
         [min_, max_],
         "b--",
         label="base")
plt.plot([lidar_data.coords["rangebin"][indx_top]] * 2,
         [min_, max_],
         "y--",
         label="top")

plt.plot([original_data.coords["rangebin"][indx_base_diego]] * 2,
         [min_, max_],
         "g--",
         label="base diego")
plt.plot([original_data.coords["rangebin"][indx_top_diego]] * 2,
         [min_, max_],
         "k--",
         label="top diego")

plt.ylabel("RCS")
plt.xlabel("Altitude (m)")
plt.legend()
plt.grid()
plt.show()
