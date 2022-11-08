from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, atmospheric_interpolation, dead_time_correction
from lidarpy.data.pre_processor import pre_processor
from lidarpy.inversion.raman_transmittance import GetCod
from lidarpy.inversion.transmittance import get_cod as get_cod2
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
    )


ds = (lidar_data
      .isel(channel=[1, 3])
      .pipe(pre_processor, 300, process, True)
      .isel(rangebin=slice(500, 3000)))

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
             mc_iter=300).fit()
print(cod)

tau = get_cod2(ds,
               [11767.5, 15262.5],
               355,
               pressure,
               temperature)

print(tau)
