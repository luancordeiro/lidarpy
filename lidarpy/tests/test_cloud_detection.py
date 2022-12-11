from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, get_uncertainty, dead_time_correction
import os
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from lidarpy.clouds.cloud_detection import CloudFinder

open_diego_data = False

if open_diego_data:
    df = pd.read_csv("data/cloud_detection_tester.txt")
    lidar_data = xr.Dataset({"phy": xr.DataArray(df["2"], dims=["rangebin"], coords=[df["1"]])})
    sigma = df["3"].to_numpy()
else:
    directory = "data/binary"
    files = [file for file in os.listdir(directory) if file.startswith("RM")][:15]
    lidar_data = GetData(directory, files).get_xarray()

    def process(ds):
        return (ds
                .pipe(remove_background, [122_000, 125_000])
                .pipe(dead_time_correction, 0.004)
                .mean("time")
                .pipe(get_uncertainty, 355, 600 * len(files)))

    lidar_data = lidar_data.pipe(process)

    lidar_data = lidar_data.sel(channel="355_1", rangebin=slice(7.5, 30_001))

jdz = 735036.004918982

print(lidar_data.phy.shape)
print(lidar_data.coords["rangebin"].shape)
plt.plot(lidar_data.coords["rangebin"], lidar_data.phy * lidar_data.coords["rangebin"] ** 2, "--")
plt.ylabel("RCS")
plt.xlabel("Altitude (m)")
plt.grid()
plt.show()
#
plt.plot(lidar_data.coords["rangebin"], lidar_data.sigma.data * lidar_data.coords["rangebin"] ** 2, "--")
plt.ylabel("sigma")
plt.xlabel("Altitude (m)")
plt.grid()
plt.show()

cloud = CloudFinder(lidar_data, 355, 3000, 5, jdz)
z_base, z_top, z_max_capa, nfz_base, nfz_top, nfz_max_capa = cloud.fit()

print("z_base", z_base)
print("z_top", z_top)

rcs = (lidar_data.phy * lidar_data.coords["rangebin"] ** 2)
indx_base = lidar_data.coords["rangebin"].sel(rangebin=z_base, method="nearest").data
indx_base = lidar_data.coords["rangebin"].isin(indx_base)
indx_top = lidar_data.coords["rangebin"].sel(rangebin=z_top, method="nearest").data
indx_top = lidar_data.coords["rangebin"].isin(indx_top)


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
