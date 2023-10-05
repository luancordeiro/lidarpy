import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lidarpy.inversion import LidarRatioCalculator
from lidarpy.molecular import AlphaBetaMolecular

weak_cloud = 0

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

plt.plot(my_data[:, 0], my_data[:, 1] * my_data[:, 0] ** 2)
plt.show()

lr = LidarRatioCalculator(signal=my_data[:, 1], rangebin=my_data[:, 0], sigma=np.sqrt(my_data[:, 0]),
                          molecular_data=ds_mol, z_ref=[10e3, 14e3],
                          cloud_lims=[5850, 6160], mc=True, mc_niter=200).fit()

print(lr)
