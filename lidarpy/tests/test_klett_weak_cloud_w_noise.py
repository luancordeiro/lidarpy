import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from lidarpy.inversion.klett import Klett
from lidarpy.plot.plotter import compare_w_sol

links = ["http://lalinet.org/uploads/Analysis/Concepcion2014/ristori-bg1e0.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/ristori-bg1e2.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/ristori-bg1e4.txt",
         "http://lalinet.org/uploads/Analysis/Concepcion2014/ristori-bg1e6.txt"]

titles = [f"BG = 10^{i}" for i in range(0, 7, 2)]

df_sol = pd.read_csv("data/sol_lalinet_weak_cloud.txt", delimiter="\t")
df_sonde = pd.read_csv("data/sonde_lalinet.txt", delimiter="\t")
df_sonde = (df_sonde
            .assign(pressure=lambda x: x.pressure * 100)
            .assign(temperature=lambda x: x.temperature + 273.15)
            [["altitude", "pressure", "temperature"]])


for link, title in zip(links, titles):
    my_data = np.genfromtxt(link)
    ds = xr.DataArray(my_data[:, 1], dims=["rangebin"])
    ds.coords["rangebin"] = my_data[:, 0]
    print(ds.shape)
    ds = xr.Dataset({"phy": ds})

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
    plt.title(title)
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
    compare_w_sol(ds.coords["rangebin"].data[ind],
                  alpha[ind],
                  df_sol["z"].to_numpy()[ind],
                  df_sol["alpha-cld"].to_numpy()[ind],
                  0)
