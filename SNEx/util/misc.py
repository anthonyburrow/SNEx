from ..util.constants import c


def check_method(method, method_list):
    method = method.lower()

    if method is None:
        method = method_list[0]
    elif method not in method_list:
        raise RuntimeError(f'"{method}" method is not recognized.')

    return method


def wave_to_freq(wave):
    return c * 1.e8 / wave
