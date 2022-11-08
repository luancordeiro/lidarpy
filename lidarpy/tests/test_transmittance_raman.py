from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, atmospheric_interpolation, dead_time_correction, groupby_nbins
from lidarpy.data.pre_processor import pre_processor
from lidarpy.inversion.raman_transmittance import GetCod
from lidarpy.inversion.transmittance import GetCod as GetCod2
import os
import matplotlib.pyplot as plt
import pandas as pd

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")]

lidar_data = (GetData(directory, files[:30])
              .get_xarray())


def process(lidar_data_):
    return (
        lidar_data_
        .pipe(remove_background, [lidar_data.coords["rangebin"][-1] - 3000, lidar_data.coords["rangebin"][-1]])
        .pipe(dead_time_correction, 0.004)
        .mean("time")
        .rolling(rangebin=9, center=True)
        .mean()
    )


ds = (lidar_data
      .isel(channel=[1, 3])
      # .pipe(pre_processor, 500, process, True)
      .pipe(process)
      .isel(rangebin=slice(500, 2500)))

ds.phy.sel(rangebin=slice(11767.5, 15262.5)).plot(col="channel")
plt.show()

temperature, pressure = atmospheric_interpolation(ds.coords["rangebin"].data,
                                                  pd.read_csv("data/sonde_data.txt"))

cod = GetCod(ds,
             [11767.5, 15262.5],
             355,
             387,
             0,
             pressure,
             temperature,
             mc_iter=None).fit()
print(cod)

tau = GetCod2(ds,
              [11767.5, 15262.5],
              355,
              pressure,
              temperature,
              mc_iter=None).fit()
print(tau)
