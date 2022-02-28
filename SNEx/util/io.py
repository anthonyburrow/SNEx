from numpy import loadtxt


def read_spectrum(filename, wave_range=None):
    data = loadtxt(filename)

    if wave_range is not None:
        wave_mask = (wave_range[0] < data[:, 0]) & (data[:, 0] < wave_range[1])
        data = data[wave_mask]

    # Deredden

    return data
