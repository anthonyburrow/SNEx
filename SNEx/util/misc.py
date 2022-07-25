import os
import shutil

from ..util.constants import c


def check_method(method, method_list):
    if method is None:
        method = method_list[0]

    method = method.lower()

    if method not in method_list:
        raise RuntimeError(f'"{method}" method is not recognized.')

    return method


def wave_to_freq(wave):
    return c * 1.e8 / wave


def between_mask(wavelengths, wave_range):
    return (wave_range[0] <= wavelengths) & (wavelengths <= wave_range[1])


def prune_data(data, wave_range=None):
    if wave_range is None:
        return data

    wave_mask = between_mask(data[:, 0], wave_range)
    return data[wave_mask]


def setup_clean_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return

    for fn in os.listdir(directory):
        fp = os.path.join(directory, fn)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.unlink(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
        except Exception as e:
            print(f'Failed to delete {fp}. Reason: {e}')


def get_normalization(flux, norm_method=None, *args, **kwargs):
    if norm_method == 'mean' or norm_method is None:
        norm = flux.mean(axis=-1)
    if norm_method == 'median':
        norm = np.median(flux, axis=-1)
    elif norm_method == 'max':
        norm = flux.max(axis=-1)

    return norm
