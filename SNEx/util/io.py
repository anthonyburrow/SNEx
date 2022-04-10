from numpy import loadtxt
from spextractor.util.preprocessing import preprocess


def read_spectrum(filename, *args, **kwargs):
    data = loadtxt(filename)
    data = preprocess(data, *args, **kwargs)

    return data
