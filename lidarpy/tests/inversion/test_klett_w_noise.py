import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lidarpy.inversion import Klett
from lidarpy.molecular import AlphaBetaMolecular

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

df_sonde = pd.read_csv("data/sonde_lalinet.txt", delimiter="\t")
df_sonde = (df_sonde
            .assign(pressure=lambda x: x.pressure * 100)
            .assign(temperature=lambda x: x.temperature + 273.15)
            [["altitude", "pressure", "temperature"]])

ds_mol = AlphaBetaMolecular(df_sonde['pressure'].to_numpy(), df_sonde['temperature'].to_numpy(),
                            df_sonde['altitude'].to_numpy(), 355).get_params()

for title, link in zip(titles, links):
    my_data = np.genfromtxt(link)

    my_data[:, 1] = my_data[:, 1] - my_data[:, 1][-100:].mean()

    plt.figure(figsize=(9, 4))
    plt.plot(my_data[:, 0], my_data[:, 1] * my_data[:, 0] ** 2)
    indx = (my_data[:, 0] > 9000) & (my_data[:, 0] < 15000)
    plt.plot(my_data[:, 0][indx],
             (my_data[:, 1] * my_data[:, 0] ** 2)[indx],
             "*",
             color="red",
             label="reference region")
    plt.legend()
    plt.title(title)
    plt.ylabel("RCS")
    plt.xlabel("altitude (m)")
    plt.grid()
    plt.show()

    klett = Klett(signal=my_data[:, 1], rangebin=my_data[:, 0], molecular_data=ds_mol,
                  z_ref=[9e3, 15e3], lidar_ratio=28)

    alpha, beta, lr = klett.fit()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    ax.plot(klett.rangebin, alpha, '-', linewidth=2, color='blue', label='computed')
    ax.plot(df_sol["altitude"], df_sol["particle_extinction_coefficient"], '--', color='red', label='solution')
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.show()
