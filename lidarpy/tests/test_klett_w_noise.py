import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from lidarpy.inversion.klett import Klett
from lidarpy.plot.plotter import compare_w_sol

links = ["http://lalinet.org/uploads/Analysis/Concepcion2014/holger-poisson-S1k-bg1e0.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/holger-poisson-S1k-bg1e1.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/holger-poisson-S1k-bg1e2.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/holger-poisson-S1k-bg1e3.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/holger-poisson-S1k-bg1e4.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/holger-poisson-S1k-bg1e5.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/holger-poisson-S1k-bg1e6.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/holger-poisson-S1k-bg1e7.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/holger-poisson-S1k-bg1e8.txt"]

titles = [f"BG=10^{i}" for i in range(9)]

df_sol = pd.read_csv("data/355_lalinet_solution.txt", delimiter="\t")
df_sol = (df_sol
          .assign(Pressure=lambda x: x["Pressure"] * 100)
          .assign(temperature=lambda x: x["temperature"] + 273.15))
print(df_sol.head())
print(df_sol[["particle_extinction_coefficient"]].describe())

df_sonde = pd.read_csv("data/sonde_lalinet.txt", delimiter="\t")
df_sonde = (df_sonde
            .assign(pressure=lambda x: x.pressure * 100)
            .assign(temperature=lambda x: x.temperature + 273.15)
            [["altitude", "pressure", "temperature"]])

for title, link in zip(titles, links):
    my_data = np.genfromtxt(link)

    ds = xr.DataArray(my_data[:, 1], dims=["altitude"])
    ds.coords["altitude"] = my_data[:, 0]

    plt.figure(figsize=(12, 7))
    plt.plot(ds.coords["altitude"].data, ds.data * ds.coords["altitude"].data ** 2)
    indx = (ds.coords["altitude"].data > 9000) & (ds.coords["altitude"].data < 15000)
    plt.plot(ds.coords["altitude"].data[indx],
             (ds.data * ds.coords["altitude"].data ** 2)[indx],
             "*",
             color="red",
             label="reference region")
    plt.legend()
    plt.title(title)
    plt.ylabel("RCS")
    plt.xlabel("altitude (m)")
    plt.grid()
    plt.show()

    klett = Klett(ds,
                  355,
                  df_sonde["pressure"].to_numpy(),
                  df_sonde["temperature"].to_numpy(),
                  [9000, 15000],
                  28)

    alpha, beta, lr = klett.fit()

    compare_w_sol(ds.coords["altitude"],
                  alpha,
                  df_sol["altitude"],
                  df_sol["particle_extinction_coefficient"],
                  0)
