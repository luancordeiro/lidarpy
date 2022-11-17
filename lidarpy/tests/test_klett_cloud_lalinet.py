import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from lidarpy.inversion.klett import Klett
from lidarpy.plot.plotter import plot_3graph_std, compare_w_sol

weak_cloud = 1

link = ["http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500.txt",
        "http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500_v2.txt"]

if weak_cloud:
    df_sol = pd.read_csv("data/sol_lalinet_weak_cloud.txt", delimiter="\t")

my_data = np.genfromtxt(link[weak_cloud])

ds = xr.DataArray(my_data[:, 1], dims=["rangebin"])
ds.coords["rangebin"] = my_data[:, 0]
ds = xr.Dataset({"phy": ds})

print(ds.phy.shape)

df_sonde = pd.read_csv("data/sonde_lalinet.txt", delimiter="\t")

df_sonde = (df_sonde
            .assign(pressure=lambda x: x.pressure * 100)
            .assign(temperature=lambda x: x.temperature + 273.15)
            [["altitude", "pressure", "temperature"]])

print()
print(df_sonde.head())
print()

plt.figure(figsize=(12, 7))
plt.plot(ds.coords["rangebin"].data, ds.phy.data * ds.coords["rangebin"].data ** 2)
indx = (ds.coords["rangebin"].data > 6500) & (ds.coords["rangebin"].data < 14000)
plt.plot(ds.coords["rangebin"].data[indx],
         (ds.phy.data * ds.coords["rangebin"].data ** 2)[indx],
         "*",
         color="red",
         label="reference region")
plt.legend()
plt.ylabel("RCS")
plt.xlabel("altitude (m)")
plt.grid()
plt.show()

klett = Klett(ds,
              355,
              df_sonde["pressure"].to_numpy(),
              df_sonde["temperature"].to_numpy(),
              [6500, 14000],
              28)

alpha, beta, lr = klett.fit()

ind = (ds.coords["rangebin"] > 4000) & (ds.coords["rangebin"] < 8000)

plot_3graph_std(ds.coords["rangebin"][ind],
                alpha[ind],
                beta[ind],
                lr * np.ones(alpha.shape)[ind])

AOD = cumtrapz(alpha[ind],
               ds.coords["rangebin"][ind],
               initial=0)

plt.figure(figsize=(12, 7))
plt.plot(ds.coords["rangebin"][ind], AOD)
plt.title(f"AOD[-1]={AOD[-1].round(3)}")
plt.ylabel("AOD")
plt.xlabel("Altitude (m)")
plt.grid()
plt.show()

if weak_cloud:
    compare_w_sol(ds.coords["rangebin"].data[ind],
                  alpha[ind],
                  df_sol["z"].to_numpy()[ind],
                  df_sol["alpha-cld"].to_numpy()[ind],
                  0)

    compare_w_sol(ds.coords["rangebin"].data[ind],
                  beta[ind],
                  df_sol["z"].to_numpy()[ind],
                  df_sol["beta-cld"].to_numpy()[ind],
                  1)

print(np.trapz(y=alpha[ind], x=klett.rangebin[ind]))

print(np.trapz(y=beta[ind], x=klett.rangebin[ind]))
