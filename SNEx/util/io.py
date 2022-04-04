from numpy import loadtxt
from snpy.utils.deredden import unred


def read_spectrum(filename, z=None, wave_range=None,
                  host_EBV=None, host_RV=None, MW_EBV=None, MW_RV=3.1):
    data = loadtxt(filename)

    # Deredshift
    if z is not None:
        data[:, 0] /= z + 1.

    # Use data only inside wave range
    if wave_range is not None:
        wave_mask = (wave_range[0] < data[:, 0]) & (data[:, 0] < wave_range[1])
        data = data[wave_mask]

    # MW extinction
    if MW_EBV is not None and MW_EBV != 0. and MW_RV is not None:
        data[:, 1], _a, _b = unred(data[:, 0], data[:, 1], MW_EBV, R_V=MW_RV)

    # Host extinction
    if host_EBV is not None and host_EBV != 0. and host_RV is not None:
        data[:, 1], _a, _b = unred(data[:, 0], data[:, 1], host_EBV, R_V=host_RV)

    return data
