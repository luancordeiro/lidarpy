import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lidarpy.inversion.klett import Klett
from lidarpy.plot.plotter import plot_3graph_std

# link = "http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500.txt"
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

indx_tau = (ds.coords["altitude"].data > 5700) & (ds.coords["altitude"].data < 6300)

plt.figure(figsize=(12, 7))
plt.plot(ds.coords["altitude"].data, ds.data * ds.coords["altitude"].data ** 2)
plt.plot(ds.coords["altitude"].data[indx_tau],
         (ds.data * ds.coords["altitude"].data ** 2)[indx_tau],
         "*",
         color="red",
         label="tau region")
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
              mc_iter=200,
              tau_ind=indx_tau,
              z_lims=[5800, 6150])

alpha, alpha_std, beta, beta_std, lr, tau, tau_std = klett.fit()

ind = (ds.coords["altitude"] > 4000) & (ds.coords["altitude"] < 8000)

plot_3graph_std(ds.coords["altitude"][ind],
                alpha[ind],
                beta[ind],
                lr * np.ones(alpha.shape)[ind],
                alpha_std[ind],
                beta_std[ind],
                lr * np.zeros(alpha.shape)[ind])

plt.errorbar(ds.coords["altitude"][ind],
             alpha[ind],
             alpha_std[ind])
plt.show()

print("----")
print(f"{tau.round(3)} +- {tau_std.round(3)}")
