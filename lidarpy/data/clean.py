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
