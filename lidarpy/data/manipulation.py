"""Conjunto de funções que tem como objetivo limpar os dados. Essas funções podem ser aplicados aos dados
utilizando o método .pipe() de um xarray"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular


def remove_background(ds, alt_ref: list):
    """
    Remove o ruído molecular de fundo de acordo com uma altura de referência

    ===========================================================================

    Exemplo:
    ds_clean = ds.pipe(remove_background, [30_000, 50_000])
    """

    jump = ds.coords["altitude"].data[1] - ds.coords["altitude"].data[0]

    background = (ds
                  .sel(altitude=np.arange(alt_ref[0], alt_ref[1], jump),
                       method="nearest")
                  .mean("altitude"))

    return ds - background


def groupby_nbins(ds, n_bins):
    """Agrupa o ds a cada n_bins

    O código está meio feio, mas aparentemente funciona
    """
    if n_bins in [0, 1]:
        return ds

    alt = ds.coords["altitude"].data

    return (ds
            .assign_coords(altitude=np.arange(len(alt)) // n_bins)
            .groupby("altitude")
            .sum()
            .assign_coords(altitude=lambda x: [alt[i * n_bins:(i + 1) * n_bins].mean() for i in range(len(x.altitude))])
            )


def atmospheric_interpolation(z, df_sonde):
    """Recebe um df com os dados de pressão temperature e altura e interpola para um z arbitrário"""
    f_temp = interp1d(df_sonde["alt"].to_numpy(), df_sonde["temp"].to_numpy())
    f_pres = interp1d(df_sonde["alt"].to_numpy(), df_sonde["pres"].to_numpy())

    temperature = f_temp(z)
    pressure = f_pres(z)

    return temperature, pressure


def molecular_model(lidar_data, wavelength, p_air, t_air, z_ref, co2ppmv=392, pc=True):
    if "wavelength" in lidar_data.dims:
        signal = lidar_data.sel(wavelength=f"{wavelength}_{int(pc)}").data
    else:
        signal = lidar_data.data

    z = lidar_data.coords["altitude"].data

    alpha_beta_mol = AlphaBetaMolecular(p_air, t_air, wavelength, co2ppmv)
    alpha_mol, beta_mol, _ = alpha_beta_mol.get_params()

    model = beta_mol * np.exp(-2 * cumtrapz(z, alpha_mol, initial=0)) / z ** 2

    ref = lidar_data.coords["altitude"].sel(altitude=z_ref, method="nearest").data
    ref = np.where((z == ref[0]) | (z == ref[1]))[0]

    reg = np.polyfit(model[ref[0]:ref[1]],
                     signal[ref[0]:ref[1]],
                     1)

    return reg[0] * model + reg[1]
