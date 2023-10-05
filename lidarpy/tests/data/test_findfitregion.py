import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lidarpy.data.signal_operations import FindFitRegion
from lidarpy.molecular import AlphaBetaMolecular

weak_cloud = 1

link = ["http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500.txt",
        "http://lalinet.org/uploads/Analysis/Concepcion2014/SynthProf_cld6km_abl1500_v2.txt"]

df_sol_weak = pd.read_csv("../inversion/data/sol_lalinet_weak_cloud.txt", delimiter="\t")

my_data = np.genfromtxt(link[weak_cloud])

df_sonde = pd.read_csv("../inversion/data/sonde_lalinet.txt", delimiter="\t")

df_sonde = (df_sonde
            .assign(pressure=lambda x: x.pressure * 100)
            .assign(temperature=lambda x: x.temperature + 273.15)
            [["altitude", "pressure", "temperature"]])

ds_mol = AlphaBetaMolecular(df_sonde['pressure'].to_numpy(), df_sonde['temperature'].to_numpy(),
                            df_sonde['altitude'].to_numpy(), 355).get_params()

min_, max_ = 4e3, 5.8e3 + 2e3
indx = (my_data[:, 0] >= min_) & (my_data[:, 0] <= max_)
plt.plot(my_data[:, 0][indx],
         ((my_data[:, 1] - my_data[:, 1][-200:].mean()) * my_data[:, 0] ** 2)[indx],
         '-')

finder = FindFitRegion(signal=my_data[:, 1] - my_data[:, 1][-200:].mean(),
                       sigma=np.sqrt(my_data[:, 1] - my_data[:, 1][-200:].mean()),
                       rangebin=my_data[:, 0],
                       molecular_data=ds_mol,
                       z_ref=[4e3, 5.8e3])

ref = finder.fit()

print(ref, my_data[:, 0][ref])
