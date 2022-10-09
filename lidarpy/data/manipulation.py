"""Conjunto de funções que tem como objetivo limpar os dados. Essas funções podem ser aplicados aos dados
utilizando o método .pipe() de um xarray"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from lidarpy.data.alpha_beta_mol import AlphaBetaMolecular
import matplotlib.pyplot as plt
import xarray as xr


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


def molecular_model(lidar_data, wavelength, p_air, t_air, alt_ref, co2ppmv=392, pc=True):
    if "wavelength" in lidar_data.dims:
        signal = lidar_data.sel(wavelength=f"{wavelength}_{int(pc)}").data
    else:
        signal = lidar_data.data

    alpha_beta_mol = AlphaBetaMolecular(p_air, t_air, wavelength, co2ppmv)
    alpha_mol, beta_mol, _ = alpha_beta_mol.get_params()

    z = lidar_data.coords["altitude"].data
    model = beta_mol * np.exp(-2 * cumtrapz(z, alpha_mol, initial=0)) / z ** 2

    ref = lidar_data.coords["altitude"].sel(altitude=alt_ref, method="nearest").data
    ref = np.where((z == ref[0]) | (z == ref[1]))[0]

    reg = np.polyfit(np.log(model[ref[0]:ref[1]]),
                     np.log(signal[ref[0]:ref[1]]),
                     1)

    return np.exp(reg[0] * np.log(model) + reg[1])


def remove_background_fit(lidar_data, wavelength, p_air, t_air, alt_ref, co2ppmv=392, pc=True) -> xr.DataArray:
    data = lidar_data.copy()
    if "wavelength" in lidar_data.dims:
        signal = data.sel(wavelength=f"{wavelength}_{int(pc)}").data
    else:
        signal = data.data

    alpha_beta_mol = AlphaBetaMolecular(p_air, t_air, wavelength, co2ppmv)
    alpha_mol, beta_mol, _ = alpha_beta_mol.get_params()

    z = lidar_data.coords["altitude"].data
    model = beta_mol * np.exp(-2 * cumtrapz(z, alpha_mol, initial=0)) / z ** 2

    ref = data.coords["altitude"].sel(altitude=alt_ref, method="nearest").data
    ref = np.where((z == ref[0]) | (z == ref[1]))[0]

    reg = np.polyfit(model[ref[0]:ref[1]],
                     signal[ref[0]:ref[1]],
                     1)

    signal -= reg[1]

    data.data = signal

    return data


def smooth(signal, window):
    if window % 2 == 0:
        raise Exception("Window value must be odd")
    out0 = np.convolve(signal, np.ones(window, dtype=int), 'valid') / window
    r = np.arange(1, window - 1, 2)
    start = np.cumsum(signal[:window - 1])[::2] / r
    stop = (np.cumsum(signal[:-window:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def smooth_diego_fast(signal, p_before, p_after):
    sm = p_before + p_after + 1
    y_sm = smooth(signal, sm)
    y_sm4 = np.zeros(y_sm.shape)
    y_sm4[:-sm//2 + 1] = y_sm[sm // 2:]
    return y_sm4


def signal_smoother(signal, altitude, window):
    vec_smooth = smooth(signal, window)
    vec_aux = vec_smooth[::window]
    z_aux = altitude[::window]
    if z_aux[-1] < altitude[-1]:
        ind = np.where(altitude == z_aux[-1])[0][0]
        vec_aux = np.append(vec_aux, signal[ind:])
        z_aux = np.append(z_aux, altitude[ind:])
    func = interp1d(z_aux, vec_aux)

    plt.plot(altitude, signal * altitude ** 2, label="original")
    plt.plot(altitude, func(altitude) * altitude ** 2, label="smooth")
    plt.legend()
    plt.grid()
    plt.show()

    return func(altitude)


def z_finder(altitude: np.array, alts):
    def finder(z):
        return round((z - altitude[0]) / (altitude[1] - altitude[0]))

    if type(alts) == int:
        return finder(alts)

    indx = []
    for alt in alts:
        indx.append(finder(alt))

    return indx


def get_uncertainty(lidar_data, wavelength, bg_ref: list, nshoots: int, pc: bool = True):
    if "wavelength" in lidar_data.dims:
        signal = lidar_data.sel(wavelength=f"{wavelength}_{int(pc)}").data
    else:
        signal = lidar_data.data

    t = nshoots / 20e6
    n = t * signal * 1e-6
    index = z_finder(lidar_data.coords["altitude"].data, bg_ref)
    bg = signal[index[0]:index[1]].mean()
    n_bg = t * bg * 1e6
    sigma_n = (n + n_bg) ** 0.5

    plt.plot(lidar_data.coords["altitude"].data, sigma_n * 1e-6 / t)
    plt.show()
    return sigma_n * 1e-6 / t


def dead_time_correction(lidar_data, dead_time):
    dead_times = np.array([
        dead_time * wavelength.endswith("1") for wavelength in lidar_data.coords["wavelength"].data
    ])

    print(lidar_data.sel(wavelength="355_1").data[:10])
    lidar_data.data = lidar_data.data / (1 - dead_times @ lidar_data.data)
    print(lidar_data.sel(wavelength="355_1").data[:10])

    return lidar_data
