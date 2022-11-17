from lidarpy.data.read_binary import GetData
from lidarpy.data.manipulation import remove_background, atmospheric_interpolation, dead_time_correction, groupby_nbins, z_finder
# from lidarpy.data.pre_processor import pre_processor
from lidarpy.inversion.raman_transmittance import GetCod
from lidarpy.inversion.transmittance import GetCod as GetCod2
import os
import matplotlib.pyplot as plt
import pandas as pd
from lidarpy.inversion.raman import Raman2
from lidarpy.inversion.klett import Klett
import numpy as np

directory = "data/binary"
files = [file for file in os.listdir(directory) if file.startswith("RM")][:30]

lidar_data = GetData(directory, files).get_xarray()


def process(lidar_data_):
    return (
        lidar_data_
        .pipe(remove_background, [lidar_data.coords["rangebin"][-1] - 5000, lidar_data.coords["rangebin"][-1]])
        .pipe(dead_time_correction, 0.004)
        .mean("time")
        # .assign(phy=lambda x: x.phy * 247921)
        # .rolling(rangebin=9, center=True)
        # .mean()
    )


r1 = 11640
r2 = 15247.5

range_ = [r1, r2]

ds = (
    lidar_data
    .isel(channel=[1, 3])
    # .pipe(pre_processor, 500, process, True)
    .pipe(process)
    .isel(rangebin=slice(100, 3000))
)

# ds.phy.sel(channel="355_1").plot()
# plt.show()

ds.phy.sel(rangebin=slice(10_000, 15000)).plot(col="channel")
plt.show()

da = ds.phy.sel(channel="355_1", rangebin=slice(10_000, 16000))

plt.plot(da.coords["rangebin"], da.data * da.coords["rangebin"] ** 2, "k-")
plt.plot([r1, r1],
         [min(da.data * da.coords["rangebin"] ** 2), max(da.data * da.coords["rangebin"] ** 2)],
         "b--",
         label="base")
plt.plot([r2, r2],
         [min(da.data * da.coords["rangebin"] ** 2), max(da.data * da.coords["rangebin"] ** 2)],
         "g--",
         label="top")
plt.legend()
plt.show()

temperature, pressure = atmospheric_interpolation(ds.coords["rangebin"].data,
                                                  pd.read_csv("data/sonde_data.txt"))

cod = GetCod(ds,
             range_,
             355,
             387,
             0,
             pressure,
             temperature,
             mc_iter=None).fit()
print(f"cod raman tranmittance = {cod}")

tau = GetCod2(ds,
              [11640, 15247.5],
              355,
              pressure,
              temperature,
              mc_iter=None).fit()
print(f"cod tranmittance {tau}")

##############################

raman = Raman2(ds,
               355,
               387,
               0,
               pressure,
               temperature,
               [15247.5 + 200, 15247.5 + 2200])

alpha, beta, lr = raman.fit()

plt.plot(raman.rangebin, beta, "k-")
plt.plot([11640, 11640],
         [min(beta), max(beta)],
         "b--",
         label="base")
plt.plot([15247.5, 15247.5],
         [min(beta), max(beta)],
         "g--",
         label="top")

plt.title("Raman")

plt.show()

ind_bot = z_finder(raman.rangebin, r1)
ind_top = z_finder(raman.rangebin, r2)

beta_int = np.trapz(y=beta[ind_bot:ind_top],
                    dx=raman.rangebin[1] - raman.rangebin[0])

# beta_int = np.trapz(y=beta,
#                     x=raman.rangebin)

lidar_ratio = cod / beta_int
alpha_int = np.trapz(y=alpha[ind_bot:ind_top],
                     x=raman.rangebin[ind_bot:ind_top])

print(f"alpha integral = {alpha_int}")
print(f"beta integral = {beta_int}")
print(f"lr bulk = {lidar_ratio}")
print(f"mean lr = {np.mean(lr)}")

#######################

klett = Klett(ds,
              355,
              pressure,
              temperature,
              [15247.5 + 500, 15247.5 + 1500],
              20)

alpha, beta, lr = klett.fit()

plt.plot(klett.rangebin, alpha, "k-")
plt.plot([11640, 11640],
         [min(alpha), max(alpha)],
         "b--",
         label="base")
plt.plot([15247.5, 15247.5],
         [min(alpha), max(alpha)],
         "g--",
         label="top")
plt.ylabel("alpha")
plt.title("Klett")
plt.show()

plt.plot(klett.rangebin, beta, "k-")
plt.plot([11640, 11640],
         [min(beta), max(beta)],
         "b--",
         label="base")
plt.plot([15247.5, 15247.5],
         [min(beta), max(beta)],
         "g--",
         label="top")
plt.ylabel("beta")
plt.title("Klett")
plt.show()

print("---------")
alpha_int = np.trapz(y=alpha[ind_bot:ind_top],
                     x=raman.rangebin[ind_bot:ind_top])

beta_int = np.trapz(y=beta[ind_bot:ind_top],
                    dx=raman.rangebin[1] - raman.rangebin[0])

print(f"alpha integral = {alpha_int}")
print(f"beta integral = {beta_int}")
print(f"lr bulk = {alpha_int / beta_int}")
