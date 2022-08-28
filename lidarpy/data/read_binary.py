import re
import numpy as np
from datetime import datetime
import xarray as xr


class GetData:
    def __init__(self, directory: str, files_name: list) -> None:
        self.directory = directory
        self.files_name = files_name

    @staticmethod
    def profile_read(f_name: str) -> tuple:
        """Faz a leitura de um arquivo do tipo binário e devolve o cabeçalho e os dados"""
        with open(f_name, 'r', encoding='utf8', errors='ignore') as fp:
            #  Linha 1
            regexp = re.compile('([\w]{9}.[\d]{3})')  # filename

            line = regexp.search(fp.readline())

            head = {'file': line.group(1)}

            #  Linha 2
            regexp = re.compile(' ([\w]*) '  # site
                                '([\d]{2}/[\d]{2}/[\d]{4}) '  # datei
                                '([\d]{2}:[\d]{2}:[\d]{2}) '  # houri
                                '([\d]{2}/[\d]{2}/[\d]{4}) '  # datef
                                '([\d]{2}:[\d]{2}:[\d]{2}) '  # hourf
                                '([\d]{4}) '  # alt
                                '(-?[\d]{3}\.\d) '  # lon
                                '(-?[\d]{3}\.\d) '  # lat
                                '(-?[\d]{1,2}) '  # zen
                                '[\d]{2} '  # ---- empty
                                '([\d]{2}\.\d) '  # T0
                                '([\d]{4}\.\d)')  # P0

            line = regexp.search(fp.readline())

            head['site'] = line.group(1)
            head['datei'] = line.group(2)
            head['houri'] = line.group(3)
            head['datef'] = line.group(4)
            head['hourf'] = line.group(5)

            def date_num(d):
                return 366 + d.toordinal() + (d - datetime.fromordinal(d.toordinal())).total_seconds() / (24 * 60 * 60)

            jdi = head['datei'] + ' ' + head['houri']
            jdi_strip = datetime.strptime(jdi, '%d/%m/%Y %H:%M:%S')

            jdf = head['datef'] + ' ' + head['hourf']
            jdf_strip = datetime.strptime(jdf, '%d/%m/%Y %H:%M:%S')

            head['jdi'] = date_num(jdi_strip)
            head['jdf'] = date_num(jdf_strip)

            head['alt'] = int(line.group(6))
            head['lon'] = float(line.group(7))
            head['lat'] = float(line.group(8))
            head['zen'] = float(line.group(9))
            head['T0'] = float(line.group(10))
            head['P0'] = float(line.group(11))

            #  Linha 3
            regexp = re.compile('([\d]{7}) '  # nshoots    
                                '([\d]{4}) '  # nhz
                                '([\d]{7}) '  # nshoots2
                                '([\d]{4}) '  # nhz2
                                '([\d]{2}) ')  # nch

            line = regexp.search(fp.readline())

            head['nshoots'] = int(line.group(1))
            head['nhz'] = int(line.group(2))
            head['nshoots2'] = int(line.group(3))
            head['nhz2'] = int(line.group(4))
            head['nch'] = int(line.group(5))

            #  Canais
            head['ch'] = {}
            nch = head['nch']  # Número de canais

            regexp = re.compile('(\d) '  # active
                                '(\d) '  # photons
                                '(\d) '  # elastic
                                '([\d]{5}) '  # ndata
                                '\d '  # ----
                                '([\d]{4}) '  # pmtv
                                '(\d\.[\d]{2}) '  # binw
                                '([\d]{5})\.'  # wlen
                                '([osl]) '  # pol
                                '[0 ]{10} '  # ----
                                '([\d]{2}) '  # bits
                                '([\d]{6}) '  # nshoots
                                '(\d\.[\d]{3,4}) '  # discr
                                '([\w]{3})')  # tr

            channels = ''.join([next(fp) for _ in range(nch)])  # Aqui eu imprimo todos os canais

            lines = np.array(regexp.findall(channels))

            head['ch']['active'] = lines[:, 0].astype(int)
            head['ch']['photons'] = lines[:, 1].astype(int)
            head['ch']['elastic'] = lines[:, 2].astype(int)
            head['ch']['ndata'] = lines[:, 3].astype(int)
            head['ch']['pmtv'] = lines[:, 4].astype(int)
            head['ch']['binw'] = lines[:, 5].astype(float)
            head['ch']['wlen'] = lines[:, 6].astype(int)
            head['ch']['pol'] = lines[:, 7]
            head['ch']['bits'] = lines[:, 8].astype(int)
            head['ch']['nshoots'] = lines[:, 9].astype(int)
            head['ch']['discr'] = lines[:, 10].astype(float)
            head['ch']['tr'] = lines[:, 11]

            # Criei os arrays phy e raw antes, pois no matlab elas são criadas enquanto declaradas

            max_linhas = max(head['ch']['ndata'])  # A solucao que encontrei aqui foi achar o max de
            # linhas possivel que phy e raw podem ter para declarar antes

            phy = np.zeros((max_linhas, nch))
            raw = np.zeros((max_linhas, nch))

            # conversion factor from raw to physical units
            for ch in range(nch):
                nz = head['ch']['ndata'][ch]
                _ = np.fromfile(fp, np.byte, 2)
                tmpraw = np.fromfile(fp, np.int32, nz)

                if head['ch']['photons'][ch] == 0:
                    d_scale = head['ch']['nshoots'][ch] * (2 ** head['ch']['bits'][ch]) / (head['ch']['discr'][ch]
                                                                                           * 1e3)
                else:
                    d_scale = head['ch']['nshoots'][ch] / 20

                tmpphy = tmpraw / d_scale

                # copy to final destination
                phy[:nz, ch] = tmpphy[:nz]
                raw[:nz, ch] = tmpraw[:nz]

        return head, phy.T, raw.T

    def get_xarray(self) -> xr.DataArray:
        """Esse método tá assumindo que todas as observações foram tomadas no mesmo local e nas mesmas condições
        a única diferença é o tempo de início da medida"""
        times = []
        phys = []
        for file in self.files_name:
            head, phy, raw = self.profile_read(f"{self.directory}/{file}")

            times.append(head["jdi"])
            phys.append(phy)

        wavelengths = [f"{wavelength}_{photon}" for wavelength, photon in zip(head["ch"]["wlen"], head["ch"]["photons"])]
        phys = np.array(phys)
        alt = np.arange(1, len(phys[0][0]) + 1) * 7.5

        return xr.DataArray(phys, coords=[times, wavelengths, alt], dims=["time", "wavelength", "altitude"])

    def to_netcdf(self, directory: str = None, save_name: str = None) -> None:
        directory = f"{directory}/" if not directory.endswith("/") else directory
        lidar_data = self.get_xarray()
        lidar_data.to_netcdf(f"{directory}{save_name}.nc")
