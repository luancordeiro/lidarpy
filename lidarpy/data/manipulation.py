"""Conjunto de funções que tem como objetivo limpar os dados. Essas funções podem ser aplicados aos dados
utilizando o método .pipe() de um xarray"""
import numpy as np


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

    mean = alt[:n_bins].mean()

    print(mean)

    return (ds
            .assign_coords(altitude=np.arange(len(alt)) // n_bins)
            .groupby("altitude")
            .mean()
            .assign_coords(altitude=lambda x: [alt[i * n_bins:(i + 1) * n_bins].mean() for i in range(len(x.altitude))])
            )
