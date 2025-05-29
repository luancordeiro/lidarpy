import numpy as np
import xarray as xr
from ..utils.constants import Constants


class AlphaBetaMolecular:
    """
    Computes molecular optical properties based on atmospheric conditions.

    Given an atmospheric pressure profile, temperature profile, wavelength,
    and CO2 concentration, this class provides methods to calculate the molecular
    extinction coefficient, molecular backscattering coefficient, and molecular lidar ratio.

    Parameters:
    -----------
    p_air : np.ndarray
        Atmospheric pressure profile [P].
    t_air : np.ndarray
        Atmospheric temperature profile [K].
    wavelength : float
        Wavelength in nanometers.
    co2ppmv : float
        CO2 concentration in parts per million by volume.

    Methods:
    --------
    set_wavelength(wavelength: float) -> AlphaBetaMolecular:
        Set a new wavelength for calculations.

    get_params() -> Tuple[np.ndarray, np.ndarray, float]:
        Compute and return the molecular extinction coefficient, molecular backscattering coefficient,
        and molecular lidar ratio.

    Attributes:
    -----------
    alpha_mol : np.ndarray
        Molecular extinction coefficient [m^-1].
    beta_mol : np.ndarray
        Molecular backscattering coefficient [m^-1 sr^-1].
    lr_mol : float
        Molecular lidar ratio [sr].

    Authors:
    --------
    H. M. J. Barbosa (hbarbosa@if.usp.br), IF, USP, Brazil
    B. Barja (bbarja@gmail.com), GOAC, CMC, Cuba
    R. Costa (re.dacosta@gmail.com), IPEN, Brazil
    """

    def __init__(self, rangebin: np.ndarray, p_air: np.ndarray, t_air: np.ndarray, wavelength: float,
                 co2ppmv: float = 372):
        self.rangebin = rangebin
        self.p_air = p_air
        self.t_air = t_air
        self.wavelength = wavelength * 1e-9
        self.const = Constants(co2ppmv)

    def set_wavelength(self, wavelength):
        self.wavelength = wavelength * 1e-9
        return self

    def _refractive_index(self):
        if self.wavelength * 1e6 > 0.23:
            dn300 = (5791817 / (238.0185 - 1 / (self.wavelength * 1e6) ** 2) +
                     167909 / (57.362 - 1 / (self.wavelength * 1e6) ** 2)) * 1e-8
        else:
            dn300 = (8060.51 + 2480990 / (132.274 - 1 / (self.wavelength * 1e6) ** 2) +
                     14455.7 / (39.32957 - 1. / (self.wavelength * 1e6) ** 2)) * 1e-8

        dn_air = dn300 * (1 + (0.54 * (self.const.co2ppmv * 1e-6 - 0.0003)))
        n_air = dn_air + 1
        return n_air

    def _king_factor(self):
        f_n2 = 1.034 + (3.17e-4 / ((self.wavelength * 1e6) ** 2))
        f_o2 = 1.096 + (1.385e-3 / ((self.wavelength * 1e6) ** 2)) + (1.448e-4 / ((self.wavelength * 1e6) ** 4))
        f_ar = 1
        f_co2 = 1.15
        f_air = (self.const.N2ppv * f_n2 + self.const.O2ppv * f_o2 + self.const.Arppv * f_ar +
                 self.const.co2ppmv * 1e-6 * f_co2) / (self.const.N2ppv + self.const.O2ppv + self.const.Arppv +
                                                       self.const.co2ppmv * 1e-6)
        return f_air

    def _depolarization_ratio(self):
        f_air = self._king_factor()
        rho_air = (6 * f_air - 6) / (3 + 7 * f_air)
        return rho_air

    def _phase_function(self):
        rho_air = self._depolarization_ratio()
        gamma_air = rho_air / (2 - rho_air)
        pf_mol = 0.75 * ((1 + 3 * gamma_air) + (1 - gamma_air) * (np.cos(np.pi) ** 2)) / (1 + 2 * gamma_air)
        return pf_mol

    def _cross_section(self):
        n_air = self._refractive_index()
        f_air = self._king_factor()
        sigma_std = 24 * (np.pi ** 3) * ((n_air ** 2 - 1) ** 2) * f_air / ((self.wavelength ** 4) *
                                                                           (self.const.Nstd ** 2) *
                                                                           (((n_air ** 2) + 2) ** 2))
        return sigma_std

    def _vol_scattering_coeff(self):
        sigma_std = self._cross_section()
        alpha_std = self.const.Nstd * sigma_std
        alpha_mol = self.p_air / self.t_air * self.const.Tstd / self.const.Pstd * alpha_std
        return alpha_mol

    def _ang_vol_scattering_coeff(self, alpha_mol):
        pf_mol = self._phase_function()
        lr_mol = 4 * np.pi / pf_mol
        beta_mol = alpha_mol / lr_mol
        return beta_mol, lr_mol

    def get_params(self):
        alpha_mol = self._vol_scattering_coeff()
        beta_mol, lr_mol = self._ang_vol_scattering_coeff(alpha_mol)

        alpha_mol = xr.DataArray(alpha_mol, coords=[('rangebin', self.rangebin)], name='alpha')
        beta_mol = xr.DataArray(beta_mol, coords=[('rangebin', self.rangebin)], name='beta')
        lr_mol = xr.DataArray(lr_mol * np.ones_like(self.rangebin), coords=[('rangebin', self.rangebin)], name='lidar_ratio')

        ds = xr.merge([alpha_mol, beta_mol, lr_mol])

        return ds
