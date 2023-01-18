import re
import numpy as np
from datetime import datetime
import xarray as xr


def _compare_dictionaries(d1, d2):
    bool_ = True
    for (key1, value1), (key2, value2) in zip(d1.items(), d2.items()):
        if type(value1) == np.ndarray:
            bool_ *= (key1 == key2) & (value1 == value2).all()
        else:
            bool_ *= (key1 == key2) & (value1 == value2)
    return bool_


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

    def get_xarray(self) -> xr.Dataset:
        """Esse método tá assumindo que todas as observações foram tomadas no mesmo local e nas mesmas condições
        a única diferença é o tempo de início da medida"""
        times = []
        datei = []
        houri = []
        datef = []
        hourf = []
        jdi = []
        jdf = []
        pressures_0 = []
        temperatures_0 = []
        phys = []
        raws = []
        nshoots = []
        zenitals = []
        wavelengths_ = []
        actives = []
        photons = []
        elastic = []
        ndata = []
        pmtv = []
        binw = []
        pol = []
        bits = []
        tr = []
        discr = []
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
                first_ch = first_head["ch"].copy()
                for key in ["file", "datei", "houri", "datef", "hourf", "jdi", "jdf", "T0", "P0", "ch", "nshoots", "nshoots2", "zen", "site"]:
                    first_head.pop(key)
                length = len(phy[0,:])
            else:
                new_head = head.copy()
                for key in ["file", "datei", "houri", "datef", "hourf", "jdi", "jdf", "T0", "P0", "ch", "nshoots", "nshoots2", "zen", "site"]:
                    new_head.pop(key)
                bool1 = _compare_dictionaries(first_head, new_head)
                # bool2 = _compare_dictionaries(first_ch, new_ch)
                if not bool1:
                    raise Exception("All headers in the directory must be the same.")

            times.append((head["jdi"] + head["jdf"]) / 2)
            datei.append(head["datei"])
            houri.append(head["houri"])
            datef.append(head["datef"])
            hourf.append(head["hourf"])
            jdi.append(head["jdi"])
            jdf.append(head["jdf"])
            pressures_0.append(head["P0"])
            temperatures_0.append(head["T0"])
            phys.append(phy[:, :length])
            raws.append(raw[:, :length])
            nshoots.append(head["ch"]["nshoots"])
            zenitals.append(head["zen"])
            wavelengths_.append(head["ch"]["wlen"])
            actives.append(head["ch"]["active"])
            photons.append(head["ch"]["photons"])
            elastic.append(head["ch"]["elastic"])
            ndata.append(head["ch"]["ndata"])
            pmtv.append(head["ch"]["pmtv"])
            binw.append(head["ch"]["binw"])
            pol.append(head["ch"]["pol"])
            bits.append(head["ch"]["bits"])
            tr.append(head["ch"]["tr"])
            discr.append(head["ch"]["discr"])

        print(f"problemas={count} de {len(self.files_name)}")
        wavelengths = [f"{wavelength}_{photon}"
                       for wavelength, photon
                       in zip(head["ch"]["wlen"], head["ch"]["photons"])]

        rangebin = np.arange(1, len(phys[0][0]) + 1) * 7.5
        da_phy = xr.DataArray(phys, coords=[times, wavelengths, rangebin], dims=["time", "channel", "rangebin"])
        da_raw = xr.DataArray(raws, coords=[times, wavelengths, rangebin], dims=["time", "channel", "rangebin"])
        das = {"phy": da_phy, "raw": da_raw}
        vars_ = [datei, houri, datef, hourf, jdi, jdf, pressures_0, temperatures_0]
        names = ["datei", "houri", "datef", "hourf", "jdi", "jdf", "pressure0", "temperature0"]
        for name, var in zip(names, vars_):
            das[name] = xr.DataArray(var, coords=[times], dims="time")
        ds = xr.Dataset(das)
        ds = ds.assign(nshoots=xr.DataArray(nshoots, coords=[times, wavelengths], dims=["time", "channel"]))
        ds = ds.assign(zenital=xr.DataArray(zenitals, coords={"time": times}, dims=["time"]))
        ds = ds.assign(wavelength=xr.DataArray(wavelengths_, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))
        ds = ds.assign(active=xr.DataArray(actives, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))
        ds = ds.assign(photon=xr.DataArray(photons, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))
        ds = ds.assign(elastic=xr.DataArray(elastic, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))
        ds = ds.assign(ndata=xr.DataArray(ndata, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))
        ds = ds.assign(pmtv=xr.DataArray(pmtv, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))
        ds = ds.assign(binw=xr.DataArray(binw, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))
        ds = ds.assign(pol=xr.DataArray(pol, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))
        ds = ds.assign(bits=xr.DataArray(bits, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))
        ds = ds.assign(tr=xr.DataArray(tr, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))
        ds = ds.assign(discr=xr.DataArray(discr, coords={"time": times, "channel": wavelengths}, dims=["time", "channel"]))

        first_head["ch"] = first_ch
        ds.attrs = first_head
        ds = ds.assign(rcs=lambda x: ds.phy * ds.coords["rangebin"].data ** 2)
        return ds

    def to_netcdf(self, directory: str = None, save_name: str = None) -> None:
        directory = f"{directory}/" if not directory.endswith("/") else directory
        lidar_data = self.get_xarray()
        lidar_data.to_netcdf(f"{directory}{save_name}.nc")
