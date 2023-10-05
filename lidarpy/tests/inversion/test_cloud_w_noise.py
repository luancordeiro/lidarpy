import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lidarpy.inversion import Klett
from lidarpy.molecular import AlphaBetaMolecular

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

ds_mol = AlphaBetaMolecular(df_sonde['pressure'].to_numpy(), df_sonde['temperature'].to_numpy(),
                            df_sonde['altitude'].to_numpy(), 355).get_params()

for link, title in zip(links, titles):
    my_data = np.genfromtxt(link)

    my_data[:, 1] = my_data[:, 1] - my_data[:, 1][-50:].mean()

    klett = Klett(signal=my_data[:, 1], rangebin=my_data[:, 0], molecular_data=ds_mol,
                  z_ref=[6500, 14000], lidar_ratio=28, correct_noise=True)

    inversion = klett.fit()

    inversion2 = klett.set_correction(False).fit()

    print()

    ind = (klett.rangebin > 4000) & (klett.rangebin < 8000)
    curves = [(inversion[0][ind], inversion2[0][ind], df_sol['alpha-cld'][ind]),
              (inversion[1][ind], inversion2[1][ind], df_sol['beta-cld'][ind])]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for i, ax in enumerate(axes):
        ax.plot(klett.rangebin[ind], curves[i][0], '-', linewidth=2, color='blue', label='computed1')
        ax.plot(klett.rangebin[ind], curves[i][1], '.', linewidth=2, color='green', label='computed2')
        ax.plot(klett.rangebin[ind], curves[i][2], '--', color='red', label='solution')
        ax.legend()
        ax.grid(True)
    plt.title(title)

    plt.tight_layout()
    plt.show()
