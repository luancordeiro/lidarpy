import re
import numpy as np
from datetime import datetime
import xarray as xr


class GetData:
    def __init__(self, directory: str, files_name: list) -> None:
        self.directory = directory
        self.files_name = files_name.copy()
        self.files_name.sort()

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

    def get_xarray(self):
        """Esse método tá assumindo que todas as observações foram tomadas no mesmo local e nas mesmas condições
        a única diferença é o tempo de início da medida"""
        (times, datei, houri, datef, hourf, jdi, jdf, pressures_0, temperatures_0, phys, raws, nshoots, zen,
         wavelengths_, actives, photons, elastic, ndata, pmtv, binw, pol, bits, tr, discr) = ([], [], [], [], [], [],
                                                                                              [], [], [], [], [], [],
                                                                                              [], [], [], [], [], [],
                                                                                              [], [], [], [], [], [])
        count = 0
        length = None
        first_head = None
        for file in self.files_name:
            try:
                head, phy, raw = self.profile_read(f"{self.directory}/{file}")
            except:
                count += 1
                if count == len(self.files_name):
                    print(f"problemas={count} de {len(self.files_name)}")
                    return None
                continue

            if first_head is None:
                first_head = head.copy()
                for key in ["file", "datei", "houri", "datef", "hourf", "jdi", "jdf", "T0", "P0", "ch", "nshoots",
                            "nshoots2", "zen", "site"]:
                    first_head.pop(key)
                length = len(phy[0, :])
            else:
                new_head = head.copy()
                for key in ["file", "datei", "houri", "datef", "hourf", "jdi", "jdf", "T0", "P0", "ch", "nshoots",
                            "nshoots2", "zen", "site"]:
                    new_head.pop(key)
                bool1 = first_head == new_head
                if not bool1:
                    raise Exception("All headers in the directory must be the same.")
            phys.append(phy[:, :length])
            raws.append(raw[:, :length])
            times.append((head["jdi"] + head["jdf"]) / 2)

            vecs1 = [datei, houri, datef, hourf, jdi, jdf, pressures_0, temperatures_0, zen]
            keys1 = ["datei", "houri", "datef", "hourf", "jdi", "jdf", "P0", "T0", "zen"]
            for vec, key in zip(vecs1, keys1):
                vec.append(head[key])

            vecs2 = [nshoots, wavelengths_, actives, photons, elastic, ndata, pmtv, binw, pol, bits, tr, discr]
            keys2 = ["nshoots", "wlen", "active", "photons", "elastic", "ndata", "pmtv", "binw", "pol", "bits", "tr",
                     "discr"]
            for vec, key in zip(vecs2, keys2):
                vec.append(head["ch"][key])

        print(f"{count} perfis não aberto de {len(self.files_name)}")

        wavelengths = [f"{wavelength}_{photon}"
                       for wavelength, photon
                       in zip(head["ch"]["wlen"], head["ch"]["photons"])]

        rangebin = np.arange(1, len(phys[0][0]) + 1) * head["ch"]["binw"][0]

        das = {"phy": xr.DataArray(phys, coords=[times, wavelengths, rangebin], dims=["time", "channel", "rangebin"]),
               "raw": xr.DataArray(raws, coords=[times, wavelengths, rangebin], dims=["time", "channel", "rangebin"])}

        vars1 = [datei, houri, datef, hourf, jdi, jdf, pressures_0, temperatures_0, zen]
        names1 = ["datei", "houri", "datef", "hourf", "jdi", "jdf", "pressure0", "temperature0", "zen"]
        for name1, var1 in zip(names1, vars1):
            das[name1] = xr.DataArray(var1, coords=[times], dims="time")

        vars2 = [nshoots, wavelengths_, actives, photons, elastic, ndata, pmtv, binw, pol, bits, tr, discr]
        names2 = ["nshoots", "wlen", "active", "photon", "elastic", "ndata", "pmtv", "binw", "pol", "bits",
                  "tr", "discr"]
        for name2, var2 in zip(names2, vars2):
            das[name2] = xr.DataArray(var2, coords=[times, wavelengths], dims=["time", "channel"])
        ds = xr.Dataset(das)

        first_head["site"] = head["site"]

        ds.attrs = first_head

        ds = ds.assign(rcs=lambda x: ds.phy * ds.coords["rangebin"].data ** 2)
        return ds

    def to_netcdf(self, directory: str = None, save_name: str = None) -> None:
        directory = f"{directory}/" if not directory.endswith("/") else directory
        lidar_data = self.get_xarray()
        lidar_data.to_netcdf(f"{directory}{save_name}.nc")
