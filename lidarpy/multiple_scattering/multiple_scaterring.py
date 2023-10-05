import subprocess
import os
import random
import numpy as np
import datetime


def smooth(signal: np.array, window: int):
    if window % 2 == 0:
        raise Exception("Window value must be odd")
    out0 = np.convolve(signal, np.ones(window, dtype=int), 'valid') / window
    r = np.arange(1, window - 1, 2)
    start = np.cumsum(signal[:window - 1])[::2] / r
    stop = (np.cumsum(signal[:-window:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def multiscatter_corr_fun(hlow, height_lidar, zh1, zbase, ztope, Alpha_Klett, alpha_mol_in, Tair_in, lambda_ELASTICO, rho):
    if len(Alpha_Klett.shape) == 1:
        Alpha_Klett = Alpha_Klett.reshape(-1, 1)
        alpha_mol_in = alpha_mol_in.reshape(-1, 1)
        Tair_in = Tair_in.reshape(-1, 1)
        zbase = [[zbase]]
        ztope = [[ztope]]
    nhours = len(zbase)
    print('[1/8] directory listing finished @', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    eta_teste = np.ones(Alpha_Klett.shape)
    ratios_teste = np.ones(Alpha_Klett.shape)

    for u in range(nhours):
        if np.sum(Alpha_Klett[:, u] > 0) == 0:
            continue

        zero_one_logic = np.zeros(zh1.shape, dtype=bool)
        dz = 100
        iizb = np.where(np.array(zbase[u]) > hlow)[0]
        for j in range(len(iizb)):
            zero_one_logic[(zh1 > zbase[u][iizb[j]]-dz) & (zh1 < ztope[u][iizb[j]]+dz)] = True

        rrr = np.arange(3, len(zh1), 4)
        range_ms_teste = zh1[rrr]

        aux = np.zeros(zh1.shape)
        aux[zero_one_logic] = smooth(Alpha_Klett[zero_one_logic, u]/1000, 9)
        aux[aux < 0] = 0
        ext_ms_teste = aux[rrr]

        ext_air_ms_teste = alpha_mol_in[rrr, u]/1000

        Taux = (Tair_in[rrr, u] - 273.15)
        Taux[(Taux < -80) | (range_ms_teste > 18000)] = -80

        radius_ms_teste = (140.5010 + 5.8714 * Taux + 0.0992 * Taux ** 2 + 5.6341e-004 * Taux ** 3) * 1e-6
        options_ms = '-quiet -algorithms fast lidar'
        wavelength_ms = lambda_ELASTICO * 1e-6
        alt_MSL_ms = height_lidar
        rho_div_ms = 0.36e-3 / 2
        rho_fov_ms = np.array([(rho / 4000) / 2])  # muda a partir de 2012 4 -> 7 mm
        range_ms = range_ms_teste
        ext_ms = ext_ms_teste
        radius_ms = radius_ms_teste
        S_ms = np.array([25])
        ext_air_ms = ext_air_ms_teste
        ssa_ms = np.array([1])
        g_ms = np.array([0.8])
        ssa_air_ms = np.array([1])
        droplet_fraction_ms = np.array([0])
        pristine_ice_fraction_ms = np.array([0])

        data_ms = multiscatter(options_ms, wavelength_ms, alt_MSL_ms, rho_div_ms, rho_fov_ms, range_ms, ext_ms,
                               radius_ms, S_ms, ext_air_ms, ssa_ms, g_ms, ssa_air_ms, droplet_fraction_ms,
                               pristine_ice_fraction_ms)

        data_single = multiscatter('-algorithms single lidar', wavelength_ms, alt_MSL_ms, rho_div_ms, rho_fov_ms,
                                   range_ms, ext_ms, radius_ms, S_ms, ext_air_ms, ssa_ms, g_ms, ssa_air_ms,
                                   droplet_fraction_ms, pristine_ice_fraction_ms)

        aux_inp = 1 - np.log(data_ms["bscat"] / data_single["bscat"]) * 1 / (2 * np.cumsum(ext_ms_teste)
                                                                             * (range_ms_teste[1] - range_ms_teste[0]))

        eta_teste[:, u] = np.interp(zh1, range_ms_teste, aux_inp)
        eta_teste[np.isnan(eta_teste[:, u]), u] = 1

        aux_inp = data_single["bscat"] / data_ms["bscat"]
        ratios_teste[:, u] = np.interp(zh1, range_ms_teste, aux_inp)

    return eta_teste, ratios_teste


def multiscatter(options: str = None, wavelength: float = None, alt: float = None, rho_div: float = None,
                 rho_fov: np.ndarray = None, range_: np.ndarray = None, ext: np.ndarray = None,
                 radius: np.ndarray = None, S: np.ndarray = None, ext_air: np.ndarray = None, ssa: np.ndarray = None,
                 g: np.ndarray = None, ssa_air: np.ndarray = None, droplet_fraction: np.ndarray = None,
                 pristine_ice_fraction: np.ndarray = None, bscat_AD: np.ndarray = None,
                 bscat_air_AD: np.ndarray = None):
    """
    MULTISCATTER  Perform multiple scattering calculation for radar or lidar

     This function calculates the apparent backscatter coefficient
     accounting for multiple scattering, calling the "multiscatter" C
     code.  It can be called in a number of different ways.

     For lidar multiple scattering including both small-angle (SA) and
     wide-angle (WA) scattering:

       inversion = multiscatter(options, wavelength, alt, rho_div, rho_fov, ...
                           range, ext, radius, S, ext_air, ssa, g, ...
                           ssa_air, droplet_fraction, pristine_ice_fraction)

     where the input variables are:
       options: string containing space-separated settings for the code -
         for the default options use the empty string ''
       wavelength: instrument wavelength in m
       alt: instrument altitude in m
       rho_div: half-angle 1/e beam divergence in radians
       rho_fov: half-angle receiver field-of-view in radians (or vector of FOVs)
       range: vector of ranges at which the input inversion are located, in
          m; the first range should be the closest to the instrument
       ext: vector of extinction coefficients, in m-1
       radius: vector of equivalent-area radii, in m
       S: vector of backscatter-to-extinction ratios, in sr
       ext_air: vector of molecular extinction coefficients, in m-1
       ssa: vector of particle single-scatter albedos
       g: vector of scattering asymmetry factors
       ssa_air: vector of molecular single-scatter albedos
       droplet_fraction: vector of fractions (between 0 and 1) of the particle
          backscatter that is due to droplets - only affects small-angle
          returns
       pristine_ice_fraction: vector of fractions (between 0 and 1) of the
          particle backscatter that is due to pristine ice - only affects
          small-angle returns
       bscat_AD: adjoint input for backscatter values
       bscat_air_AD: adjoint input for the backscatter of air

     For radar multiple scattering use the same form but with options
     containing the string '-no-forward-lobe'

     If droplet_fraction and pristine_ice_fraction are omitted then they are
     assumed zero, i.e. the near-backscatter phase function is assumed
     isotropic.  If "ssa_air" is also omitted it is assumed to be 1 if
     wavelength < 1e-6 m and 0 if wavelength > 1e-6. If "ssa" and "g" are
     omitted then the WA part of the calculation is not performed. If
     "ext_air" is omitted then it is assumed to be zero.

     The "options" string contains a space-separated list of arguments
     passed directly to the executable.  Valid ones to use from matlab are:

    General options
     -single-only    Single scattering only
     -qsa-only       Don't include wide-angle multiple scattering
     -wide-only      Only wide-angle multiple scattering
     -hsrl           Output particulate and air backscatter separately
     -gaussian-receiver
                     Receiver is Gaussian rather than top-hat shaped
     -adjoint        Output the adjoint as well
     -numerical-jacobian
                     Output the Jacobian instead
     -jacobian       Approximate but faster Jacobian: only for small-angle
     -ext-only       Only calculate the Jacobian with respect to extinction

    Options for quasi-small-angle (QSA) multiple scattering calculation
     -explicit n     Use an explicit model with n orders of scattering
     -fast-qsa       Use fast O(N) QSA model
     -wide-angle-cutoff <theta>
                     Forward scattering at angles greater than <theta> radians
                     are deemed to escape, a crude way to deal with a problem
                     associated with aerosols
     -approx-exp     Appriximate the exp() function call for speed

    Options for wide-angle multiple-scattering
     -no-forward-lobe
                     Radar-like phase function behaviour: use single-scattering
                     rather than QSA
     -simple-2s-coeffts
                     Use the simple upwind Euler formulation (can be unstable
                     for high optical depth)
     -ssa-scales-forward-lobe
                     Single-scattering albedo less than unity reduces the
                     forward scattering lobe as well as wide-angle scattering
     -num-samples m  Output m samples, allowing sampling of signals appearing
                     to originate below ground

     The output is written to the structure "inversion" containing the member
     "bscat" which is the apparent backscatter coefficient. It also
     includes "range", "ext" and "radius", which are the same as the input
     values.

     If the "-adjoint" option is specified then the adjoint variables
     ext_AD, ssa_AD, g_AD and ext_bscat_ratio_AD will also be returned in
     inversion based on the input variables bscat_AD and bscat_air_AD.

     Note that this function may need to be edited to indicate where the
     "multiscatter" executable file is located.
    """
    nargin = len([var for var in locals().values() if var is not None])

    multiscatter_exec = os.path.join(os.getcwd(), "multiscatter.exe")
    in_temp = "MSCATTER_" + str(random.randint(0, 10000000)) + ".DAT"
    out_temp = "STDERR_" + str(random.randint(0, 10000000)) + ".txt"

    # Check existence of executable
    if not os.path.isfile(multiscatter_exec):
        raise Exception(f"{multiscatter_exec} not found")

    # If fewer than 9 arguments, show the help
    if nargin < 9:
        if nargin == 1 and options == "executable":
            return multiscatter_exec
        print("help multiscatter")
        return

    # Useful vectors
    z = np.zeros_like(range_)
    o = np.ones_like(range_)

    # Allow for S and radius to be a single number
    if len(S) == 1:
        S = S * o
    if len(radius) == 1:
        radius = radius * o

    # Interpret some of the options
    nfov = len(rho_fov)
    is_hsrl = "-hsrl" in options
    is_adjoint = "-adjoint" in options
    is_jacobian = "-jacobian" in options

    # Open the input file and write the comments and the first line
    with open(in_temp, 'w') as fid:
        fid.write('# This file contains an input profile of atmospheric properties for\n')
        fid.write('# the multiscatter code. The comand-line should be of the form:\n')
        fid.write('#   ./multiscatter [options] [input_file] > [output_file]\n')
        fid.write('# or\n')
        fid.write('#   ./multiscatter [options] < [input_file] > [output_file]\n')
        if options:
            fid.write(f'# Suitable options for this input file are:\n#   {options}\n')
        else:
            fid.write('# No options are necessary for this particular file.\n')
        fid.write('# The file format consists of any number of comment lines starting\n')
        fid.write('# with "#", followed by a line of 5 inversion values:\n')
        fid.write('#   1. number of points in profile\n')
        fid.write('#   2. instrument wavelength (m)\n')
        fid.write('#   3. instrument altitude (m)\n')
        fid.write('#   4. transmitter 1/e half-width (radians)\n')
        fid.write('#  5+. receiver half-widths (radians)\n')
        fid.write('#      (1/e half-width if the "-gaussian-receiver" option\n')
        fid.write('#       is specified, top-hat half-width otherwise)\n')
        fid.write('# The subsequent lines contain 4 or more elements:\n')
        fid.write('#   1. range above ground, starting with nearest point to instrument (m)\n')
        fid.write('#   2. extinction coefficient of cloud/aerosol only (m-1)\n')
        fid.write('#   3. equivalent-area particle radius of cloud/aerosol (m)\n')
        fid.write('#   4. extinction-to-backscatter ratio of cloud/aerosol (sterad)\n')
        fid.write('#   5. extinction coefficient of air (m-1) (default 0)\n')
        fid.write('#   6. single scattering albedo of cloud/aerosol\n')
        fid.write('#   7. scattering asymmetry factor of cloud/aerosol\n')
        fid.write('#   8. single scattering albedo of air (isotropic scattering)\n')
        fid.write('#   9. fraction of cloud/aerosol backscatter due to droplets\n')
        fid.write('#  10. fraction of cloud/aerosol backscatter due to pristine ice\n')
        fid.write('# Note that elements 6-8 correspond to the wide-angle\n')
        fid.write('# multiple-scattering calculation and if this is omitted then only\n')
        fid.write('# the small-angle multiple-scattering calculation is performed.\n')
        fid.write('# For more help on how to run the code, type:\n')
        fid.write('#   ./multiscatter -help\n')
        fid.write(f'{len(range_)} {wavelength} {alt} {rho_div:g}')
        for rho in rho_fov:
            fid.write(f' {rho:g}')
        fid.write('\n')

        # Set variables that may be missing
        if nargin < 15:
            pristine_ice_fraction = z
            if nargin < 14:
                droplet_fraction = z
                if nargin < 13:
                    if wavelength > 1e-6:
                        ssa_air = z
                    else:
                        ssa_air = o

        if nargin >= 12:
            # Allow g, ssa, ssa_air and *_fraction to be single values
            if len(g) == 1:
                g = g * o
            if len(ssa) == 1:
                ssa = ssa * o
            if len(ssa_air) == 1:
                ssa_air = ssa_air * o
            if len(droplet_fraction) == 1:
                droplet_fraction = droplet_fraction * o
            if len(pristine_ice_fraction) == 1:
                pristine_ice_fraction = pristine_ice_fraction * o
            if nargin > 15:  # tem problema
                # Adjoint
                if not is_hsrl:
                    bscat_air_AD = np.zeros(bscat_AD.shape)
                for ii in range(len(range_)):
                    data = [range_[ii], ext[ii], radius[ii], S[ii], ext_air[ii], ssa[ii], g[ii],
                            ssa_air[ii], droplet_fraction[ii], pristine_ice_fraction[ii],
                            *bscat_AD[ii, :], *bscat_air_AD[ii, :]]
                    line = ' '.join([str(d) for d in data]) + '\n'
                    fid.write(line)
            else:
                # Wide-angle calculation
                for z in range(len(range_)):
                    line = "{:.0f} {:.8e} {:.6f} {:.6f} {:.8e} {:.6f} {:.1f} {:.8e} {:.0f} {:.0f}\n".format(
                        range_[z], ext[z], radius[z], S[z], ext_air[z], ssa[z], g[z], ssa_air[z], droplet_fraction[z],
                        pristine_ice_fraction[z]
                    )
                    fid.write(line)
        elif nargin >= 10:  # tem problema
            # Small-angle calculation only, with specified molecular extinction
            fid.write(' '.join(map(str, [range_, ext, radius, S, ext_air])))
            fid.write('\n')
        else:  # tem problema
            # Small-angle calculation only, with no molecular extinction
            fid.write(' '.join(map(str, [range_, ext, radius, S])))
            fid.write('\n')

    # Run executable and read output
    command = [
        'powershell.exe', '-Command',
        f"Get-Content {in_temp} | & 'C:\\Users\\luant\\PycharmProjects\\lidarpy\\lidarpy\\multiple_scattering"
        f"\\multiscatter.exe' {options} 2> {out_temp}"
    ]

    # executar o comando e capturar saída
    output, error = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

    status = 0 if error.decode() == '' else 1

    if status != 0:
        print(error)
        with open(out_temp, 'r') as f:
            print(f.read())
        raise Exception('An error occurred in running the executable')

    try:
        output = output.decode()
    except Exception as e:
        print(error)
        with open(out_temp, 'r') as f:
            print(f.read())
        raise Exception('Problem interpreting output as a table of numbers')

    if not is_jacobian:
        # Create a dictionary containing the output variables
        data = {'range': [], 'ext': [], 'radius': [], 'bscat': []}
        for line in output.split('\n'):
            if line == "":
                continue
            items = line.split()
            data['range'].append(float(items[1]))
            data['ext'].append(float(items[2]))
            data['radius'].append(float(items[3]))
            data['bscat'].append(float(items[4]))

        data['range'] = np.array(data['range'])
        data['ext'] = np.array(data['ext'])
        data['radius'] = np.array(data['radius'])
        data['bscat'] = np.array(data['bscat'])

        index = 4 + nfov
        if is_hsrl:
            inversion['bscat_air'] = [[row[z] for row in output] for _ in range(index, index + nfov)]
            index += nfov
        if is_adjoint:
            inversion['ext_AD'] = [row[index + 1] for row in output]
            inversion['ssa_AD'] = [row[index + 2] for row in output]
            inversion['g_AD'] = [row[index + 3] for row in output]
            inversion['ext_bscat_ratio_AD'] = [row[index + 4] for row in output]
    else:
        n = len(range_)
        if is_hsrl:
            inversion = {'d_bscat_d_ext': [row[:len(row) // 2] for row in output[:n]],
                    'd_bscat_air_d_ext': [row[len(row) // 2:] for row in output[:n]]}
        else:
            inversion = {'d_bscat_d_ext': [row for row in output[:n]]}

        if len(output) > n:
            if is_hsrl:
                inversion['d_bscat_d_ssa'] = [row[:len(row) // 2] for row in output[n:n * 2]]
                inversion['d_bscat_air_d_ssa'] = [row[len(row) // 2:] for row in output[n:n * 2]]
                inversion['d_bscat_d_g'] = [row[:len(row) // 2] for row in output[2 * n:2 * n + n]]
                inversion['d_bscat_air_d_g'] = [row[len(row) // 2:] for row in output[2 * n:2 * n + n]]
                inversion['d_bscat_d_radius'] = [row[:len(row) // 2] for row in output[3 * n:3 * n + n]]
                inversion['d_bscat_air_d_radius'] = [row[len(row) // 2:] for row in output[3 * n:3 * n + n]]
                inversion['d_bscat_d_ext_bscat_ratio'] = output[4 * n + 1][:len(output[4 * n + 1]) // 2]
            else:
                inversion['d_bscat_d_ssa'] = output[n:n * 2, :]
                inversion['d_bscat_d_g'] = output[2 * n:2 * n + n, :]
                inversion['d_bscat_d_radius'] = output[3 * n:3 * n + n, :]
                inversion['d_bscat_d_ext_bscat_ratio'] = output[4 * n, :]

    # delete temporary files
    os.remove(in_temp)
    os.remove(out_temp)

    return data

