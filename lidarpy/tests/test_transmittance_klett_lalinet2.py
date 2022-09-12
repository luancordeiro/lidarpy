import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from lidarpy.inversion.transmittance import Transmittance
from lidarpy.inversion.klett import Klett
from lidarpy.plot.plotter import plot_3graph_std

link = "http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500_v2.txt"
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
z = ds.coords["altitude"].data
ind = (z > 5750) & (z < 10000)
plt.figure(figsize=(12, 7))
plt.plot(z[ind], (ds.data * z ** 2)[ind])
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
