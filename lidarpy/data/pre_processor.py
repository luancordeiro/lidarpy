import xarray as xr
import numpy as np


def pre_processor(lidar_data: xr.Dataset, mc_iter: int, process, pc: bool = None):
    ds = lidar_data.copy()
    if pc:
        mc_data = np.random.poisson(ds.phy.data, [mc_iter, *ds.phy.data.shape])
        processed_data = []
        for d in mc_data:
            ds.phy.data = d
            processed_data.append(ds
                                  .pipe(process)
                                  .phy
                                  .data)

        ds = ds.pipe(process)
        try:
            return (
                ds
                .assign(phy=(["rangebin", "channel"], np.mean(processed_data, axis=0)))
                .assign(sigma=(["rangebin", "channel"], np.std(processed_data, axis=0, ddof=1)))
            )
        except:
            return (
                ds
                .assign(phy=(ds.dims, np.mean(processed_data, axis=0)))
                .assign(sigma=(ds.dims, np.std(processed_data, axis=0, ddof=1)))
            )
