from scipy.optimize import curve_fit


fit_methods = ('ls',)


def least_squares(f, x, y, p0, sigma, bounds=None, *args, **kwargs):
    return curve_fit(f, x, y, p0, sigma, bounds=bounds)
