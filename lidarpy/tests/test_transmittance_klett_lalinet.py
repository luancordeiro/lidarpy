import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from lidarpy.inversion.transmittance import Transmittance
from lidarpy.inversion.klett import Klett
from lidarpy.plot.plotter import plot_3graph_std

link = "http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500.txt"
my_data = np.genfromtxt(link)

ds = xr.DataArray(my_data[:, 1], dims=["altitude"])
ds.coords["altitude"] = my_data[:, 0]
print(ds.shape)

df_sonde = pd.read_csv("data/sonde_lalinet.txt", delimiter="\t")

df_sonde = (df_sonde
            .assign(pressure=lambda x: x.pressure * 100)
            .assign(temperature=lambda x: x.temperature + 273.15)
            [["altitude", "pressure", "temperature"]])

print()
print(df_sonde.head())
print()

plt.figure(figsize=(12, 7))
plt.plot(ds.coords["altitude"].data, ds.data * ds.coords["altitude"].data ** 2)
plt.ylabel("RCS")
plt.xlabel("altitude (m)")
plt.grid()
plt.show()

tau = Transmittance(ds,
                    [5800, 6150],
                    355,
                    df_sonde["pressure"].to_numpy(),
                    df_sonde["temperature"].to_numpy()).fit(30)

print("----------------------------")
print(f"AOD_transmittance = {tau.round(2)}")
print("----------------------------")

klett = Klett(ds,
              355,
              28,
              df_sonde["pressure"].to_numpy(),
              df_sonde["temperature"].to_numpy(),
              [6500, 8000])

alpha, beta, lr = klett.fit()

ind = np.where(df_sonde
               .isin(ds.altitude.sel(altitude=[5800, 6200], method="nearest")
                     .data))[0]

plot_3graph_std(ds.coords["altitude"][ind[0]:ind[1]],
                alpha[ind[0]:ind[1]],
                beta[ind[0]:ind[1]],
                lr * np.ones(alpha.shape)[ind[0]:ind[1]])

AOD = cumtrapz(alpha[ind[0]:ind[1]],
               ds.coords["altitude"][ind[0]:ind[1]],
               initial=0)

plt.figure(figsize=(12, 7))
plt.plot(ds.coords["altitude"][ind[0]:ind[1]], AOD)
plt.title(f"AOD[-1]={AOD[-1].round(3)}")
plt.ylabel("AOD")
plt.xlabel("Altitude (m)")
plt.grid()
plt.show()
