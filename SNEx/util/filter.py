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


def boxcar(data, bins=50):
    wave = data[:, 0]
    new_wave_endpoints = np.linspace(wave[0], wave[-1], bins + 1)

    new_data = np.zeros((bins, 3))
    new_data[:, 0] = 0.5 * (new_wave_endpoints[:-1] + new_wave_endpoints[1:])

    for i in range(bins):
        wave_mask = (new_wave_endpoints[i] < wave) & \
                    (wave < new_wave_endpoints[i + 1])

        bin_data = data[wave_mask]
        new_data[i, 1] = bin_data[:, 1].mean()

        bin_flux_var = bin_data[:, 1].var()
        bin_flux_err_var = (bin_data[:, 2]**2).sum()

        new_data[i, 2] = np.sqrt(bin_flux_var + bin_flux_err_var)

    return new_data
