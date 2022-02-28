from numpy import loadtxt


def read_spectrum(filename, z=None, wave_range=None):
    data = loadtxt(filename)

    # Deredshift
    if z is not None:
        data[:, 0] /= z + 1.

    # Use data only inside wave range
    if wave_range is not None:
        wave_mask = (wave_range[0] < data[:, 0]) & (data[:, 0] < wave_range[1])
        data = data[wave_mask]

    # Deredden

    return data
