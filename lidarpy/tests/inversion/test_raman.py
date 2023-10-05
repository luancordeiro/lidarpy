import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from lidarpy.inversion import Raman
from lidarpy.molecular import AlphaBetaMolecular
from lidarpy.data.signal_operations import dead_time_correction, remove_background, groupby_nbins, get_uncertainty

df_temp_pressure = pd.read_csv("data/netcdf/earlinet_pres_temp.txt", sep=" ")
ds_solution = xr.open_dataset("data/netcdf/earlinet_solution.nc")
ds_data = xr.open_dataset("data/netcdf/earlinet_data.nc")

n_bins_mean = 0
n_bins_group = 5
alt_min = 300
alt_max = 20_000
ds_data = (ds_data
           .pipe(dead_time_correction, 0)
           .pipe(remove_background, [28_000, 30_000])
           .mean("time")
           .pipe(groupby_nbins, n_bins_group)
           # .rolling(rangebin=n_bins_mean, center=True)
           # .mean()
           .dropna("rangebin")
           .sel(rangebin=slice(alt_min, alt_max)))

df_temp_pressure = (df_temp_pressure
                    .assign(Temperature=lambda x: x.Temperature + 273.15)
                    .assign(Pressure=lambda x: x.Pressure * 100)
                    .groupby(df_temp_pressure.index // n_bins_group)
                    .mean()
                    # .rolling(n_bins_mean, center=True)
                    # .mean()
                    .dropna())

df_temp_pressure = df_temp_pressure[(df_temp_pressure.Altitude >= alt_min) & (df_temp_pressure.Altitude <= alt_max)]

alpha_beta_mol = AlphaBetaMolecular(p_air=df_temp_pressure['Pressure'],
                                    t_air=df_temp_pressure['Temperature'],
                                    rangebin=df_temp_pressure['Altitude'],
                                    wavelength=355)

ds_mol_elastic = alpha_beta_mol.get_params()
ds_mol_inelastic = alpha_beta_mol.set_wavelength(387).get_params()

raman = Raman(elastic_signal=ds_data.sel(channel='355_1').phy.data,
              inelastic_signal=ds_data.sel(channel='387_1').phy.data,
              rangebin=ds_data.rangebin.data,
              elastic_sigma=None,
              inelastic_sigma=None,
              elastic_molecular_data=ds_mol_elastic,
              inelastic_molecular_data=ds_mol_inelastic,
              lidar_wavelength=355,
              raman_wavelength=387,
              angstrom_coeff=1.8,
              raman_scatterer_numerical_density=(78.08e-2 * df_temp_pressure['Pressure']
                                                 / (1.380649e-23 * df_temp_pressure['Temperature'])).to_numpy(),
              z_ref=[10000, 12000])


def beta_smoother(raman_: Raman):
    return savgol_filter(raman_.get_beta()["elastic_aer"], 21, 2)


inversion = raman.set_beta_smooth(beta_smoother).fit()

# Criando uma figura e 3 subplots
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), sharey=True)

y = raman.rangebin
ind = (y > 500) & (y < 6e3)
y = y[ind]
axs[0].plot(ds_solution.sel(channel="355_1", rangebin=slice(500, 6e3)).extinction.data,
            ds_solution.sel(channel="355_1", rangebin=slice(500, 6e3)).rangebin.data, label="Solution")
axs[0].plot(inversion[0][ind], y, label="Computed")
axs[0].set_xlabel('Extinction')
axs[0].legend()

axs[1].plot(ds_solution.sel(channel="355_1", rangebin=slice(500, 6e3)).backscatter.data,
            ds_solution.sel(channel="355_1", rangebin=slice(500, 6e3)).rangebin.data)
axs[1].plot(inversion[1][ind], y)
axs[1].set_xlabel('Backscattering')

axs[2].plot(ds_solution.sel(channel="355_1", rangebin=slice(500, 6e3)).lidar_ratio.data,
            ds_solution.sel(channel="355_1", rangebin=slice(500, 6e3)).rangebin.data,
            label=f'Mean lidar ratio = {ds_solution.sel(channel="355_1", rangebin=slice(500, 6e3)).lidar_ratio.data.mean().round()}')
axs[2].plot(inversion[2][ind], y,
            label=inversion[2][ind].mean().round())
axs[2].set_xlabel('Lidar ratio')
axs[2].legend()

plt.tight_layout()

plt.show()
