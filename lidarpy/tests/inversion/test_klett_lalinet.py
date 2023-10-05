import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lidarpy.inversion import Klett
from lidarpy.molecular import AlphaBetaMolecular

weak_cloud = 1

link = ["http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500.txt",
        "http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500_v2.txt"]

df_sol_weak = pd.read_csv("data/sol_lalinet_weak_cloud.txt", delimiter="\t")

my_data = np.genfromtxt(link[weak_cloud])

df_sonde = pd.read_csv("data/sonde_lalinet.txt", delimiter="\t")

df_sonde = (df_sonde
            .assign(pressure=lambda x: x.pressure * 100)
            .assign(temperature=lambda x: x.temperature + 273.15)
            [["altitude", "pressure", "temperature"]])

ds_mol = AlphaBetaMolecular(df_sonde['pressure'].to_numpy(), df_sonde['temperature'].to_numpy(),
                            df_sonde['altitude'].to_numpy(), 355).get_params()

my_data[:, 1] = my_data[:, 1] - my_data[:, 1][-50:].mean()

klett = Klett(signal=my_data[:, 1], rangebin=my_data[:, 0], molecular_data=ds_mol,
              z_ref=[6500, 14000], lidar_ratio=28)

inversion = klett.fit()

print()

ind = (klett.rangebin > 4000) & (klett.rangebin < 8000)
curves = [(inversion[0][ind], df_sol_weak['alpha-cld'][ind]),
          (inversion[1][ind], df_sol_weak['beta-cld'][ind])]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
for i, ax in enumerate(axes):
    ax.plot(klett.rangebin[ind], curves[i][0], '-', linewidth=2, color='blue', label='computed')
    ax.plot(klett.rangebin[ind], curves[i][1], '--', color='red', label='solution')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
