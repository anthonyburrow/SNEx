import numpy as np


def monotonic(data, remove_unique=True):
    max_wave = data[:, 0][data[:, 1].argmax()]
    premax = data[:, 0] < max_wave

    data_filtered = data.copy()

    data_filtered[:, 1][premax] = np.maximum.accumulate(data[:, 1][premax])

    flipped_flux = np.flip(data[:, 1][~premax])
    data_filtered[:, 1][~premax] = np.flip(np.maximum.accumulate(flipped_flux))

    if remove_unique:
        _, ind = np.unique(data_filtered[:, 1], return_index=True)
        data_filtered = data_filtered[np.sort(ind)]

    return data_filtered
