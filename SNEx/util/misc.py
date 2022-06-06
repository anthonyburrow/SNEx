import os

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


def prune_data(data, wave_range=None):
    if wave_range is None:
        return data

    wave_mask = (wave_range[0] < data[:, 0]) & (data[:, 0] < wave_range[1])
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
