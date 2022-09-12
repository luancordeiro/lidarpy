from lidarpy import GetData
from lidarpy.data.manipulation import remove_background, atmospheric_interpolation
from lidarpy.inversion.transmittance import Transmittance
import os
# import matplotlib.pyplot as plt
import pandas as pd

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")]

data = GetData(directory, files)

lidar_data = data.get_xarray()

lidar_data = lidar_data.pipe(remove_background, [25_000, 80_000])

lidar_data = lidar_data.mean("time")

ds = lidar_data[[1, 3], 1100:2100].rolling(altitude=7).mean().dropna("altitude")

temperature, pressure = atmospheric_interpolation(ds.coords["altitude"].data,
                                                  pd.read_csv("data/sonde_data.txt"))

tau = Transmittance(ds, [11_700, 15_350], 355, pressure, temperature).fit()

print(f"tau = {tau}")
