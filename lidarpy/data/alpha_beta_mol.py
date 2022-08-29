import numpy as np
from lidarpy.util.constants import Constants


class AlphaBetaMolecular:
    """
    Input:

    pres - Atmospheric pressure profile [P]
    temp - Atmospheric temperature profile [K]
    lambda_ - wavelength vector [m]
    co2ppmv - CO2 concentration [ppmv]

    Output:
    alpha_mol: molecular extinction coefficient [m^-1]
    beta_mol: molecular backscattering coefficient [m^-1 sr^-1]
    lr_mol: molecular lidar ratio [sr]
    """

    def __init__(self,
                 p_air: np.ndarray,
                 t_air: np.ndarray,
                 wavelength: float,
                 co2ppmv: float):
        self.p_air = p_air
        self.t_air = t_air
        self.wavelength = wavelength
        self.const = Constants(co2ppmv)

    def set_wavelength(self, wavelength):
        self.wavelength = wavelength
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
        return alpha_mol.copy(), beta_mol.copy(), lr_mol
