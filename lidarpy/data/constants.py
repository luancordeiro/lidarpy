class Constants:
    # Atmospheric Constituents Concentration                                   [ppv]
    # ref: Seinfeld and Pandis (1998)
    N2ppv = 0.78084
    O2ppv = 0.20946
    Arppv = 0.00934
    Neppv = 1.80 * 1e-5
    Heppv = 5.20 * 1e-6
    Krppv = 1.10 * 1e-6
    H2ppv = 5.80 * 1e-7
    Xeppv = 9.00 * 1e-8

    # Atmospheric Constituents Molecular weight                                 [g mol^-1]
    # ref: Handbook of Physics and Chemistry (CRC 1997)
    N2mwt = 28.013
    O2mwt = 31.999
    Armwt = 39.948
    Nemwt = 20.18
    Hemwt = 4.003
    Krmwt = 83.8
    H2mwt = 2.016
    Xemwt = 131.29
    CO2mwt = 44.01

    # Physic Constants
    k = 1.3806503e-23   # Boltzmann Constant                                    [J K^-1]
    Na = 6.0221367e+23  # Avogadro's number                                     [# mol^-1]
    Rgas = k * Na       # Universal gas constant                                [J K^-1 mol^-1]

    # Standard Atmosphere Reference Values
    T0 = 273.15    # zero deg celcius                                           [K]
    Tstd = 288.15  # Temperature                                                [K]
    Pstd = 101325  # Pressure                                                   [Pa]

    # Bodhaine et al, 1999
    Mvol = 22.4141e-3  # Molar volume at Pstd and T0                            [m^3 mol^-1]
    Nstd = (Na / Mvol) * (T0 / Tstd)  # Molec density at Tstd and Pstd          [# m^-3]

    def __init__(self, co2ppmv=392):
        self.co2ppmv = co2ppmv
        self.CO2ppv = self.co2ppmv * 1e-6

        self.Airmwt = (self.N2ppv * self.N2mwt + self.O2ppv * self.O2mwt + self.Arppv * self.Armwt +
                       self.Neppv * self.Nemwt + self.Heppv * self.Hemwt + self.Krppv * self.Krmwt +
                       self.H2ppv * self.H2mwt + self.Xeppv * self.Xemwt + self.CO2ppv * self.CO2mwt) / \
                      (self.N2ppv + self.O2ppv + self.Arppv + self.Neppv + self.Heppv + self.Krppv +
                       self.H2ppv + self.Xeppv + self.CO2ppv)

        # Wallace and Hobbs, p. 65
        self.Rair = self.Rgas / self.Airmwt * 1e3  # Dry air gas constant       [J K^-1 kg^-1]


if __name__ == '__main__':
    c = Constants()
