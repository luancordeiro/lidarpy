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
indx = (ds.coords["altitude"].data > 6500) & (ds.coords["altitude"].data < 14000)
plt.plot(ds.coords["altitude"].data[indx],
         (ds.data * ds.coords["altitude"].data ** 2)[indx],
         "*",
         color="red",
         label="reference region")
plt.legend()
plt.ylabel("RCS")
plt.xlabel("altitude (m)")
plt.grid()
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(ds.coords["altitude"].data, ds.data * ds.coords["altitude"].data ** 2)
indx1 = (ds.coords["altitude"].data < 5800) & (ds.coords["altitude"].data > 5050)
indx2 = (ds.coords["altitude"].data > 6150) & (ds.coords["altitude"].data < 6900)
plt.plot(ds.coords["altitude"].data[indx1 | indx2],
         (ds.data * ds.coords["altitude"].data ** 2)[indx1 | indx2],
         "*",
         color="red",
         label="transmittance mean region")
plt.legend()
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
              [6500, 14000])

alpha, beta, lr = klett.fit()

ind = (ds.coords["altitude"] > 4000) & (ds.coords["altitude"] < 8000)

plot_3graph_std(ds.coords["altitude"][ind],
                alpha[ind],
                beta[ind],
                lr * np.ones(alpha.shape)[ind])

AOD = cumtrapz(alpha[ind],
               ds.coords["altitude"][ind],
               initial=0)

plt.figure(figsize=(12, 7))
plt.plot(ds.coords["altitude"][ind], AOD)
plt.title(f"AOD[-1]={AOD[-1].round(3)}")
plt.ylabel("AOD")
plt.xlabel("Altitude (m)")
plt.grid()
plt.show()
