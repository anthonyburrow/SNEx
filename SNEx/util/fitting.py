from numpy import inf
from scipy.optimize import curve_fit


fit_methods = ('ls',)


def least_squares(f, x, y, p0, sigma, bounds=None, *args, **kwargs):
    if bounds is None:
        bounds = (-inf, inf)
    return curve_fit(f, x, y, p0, sigma, bounds=bounds)
