import numpy as np


_default_interp_method = 'linear'


def _linear_interp(times, sample_flux, sample_flux_var, interp_time):
    ind = np.searchsorted(times, interp_time)

    if ind == 0:
        endpoints = [0, 1]
    elif ind == len(times):
        endpoints = [-2, -1]
    else:
        endpoints = [ind - 1, ind]

    end_times = times[endpoints]
    end_flux = sample_flux[endpoints]
    end_flux_var = sample_flux_var[endpoints]

    # Flux
    dt = end_times[1] - end_times[0]
    dt1 = interp_time - end_times[0]
    dt2 = end_times[1] - interp_time
    slope = (end_flux[1] - end_flux[0]) / dt
    interp_flux = slope * dt1 + end_flux[0]

    # Flux variance
    slope_var = end_flux_var.sum(axis=0) / dt**2

    # Extend variance for the closest endpoint
    if abs(interp_time - end_times[0]) < abs(interp_time - end_times[1]):
        interp_flux_var = dt1**2 * slope_var + end_flux_var[0]
    else:
        interp_flux_var = dt2**2 * slope_var + end_flux_var[1]

    return interp_flux, interp_flux_var


def get_interp_spectrum(times, spectra, interp_time, time_interp_method=None):
    n_samples = len(spectra)
    n_points = len(spectra[0])
    sample_flux = np.zeros((n_samples, n_points))
    sample_flux_var = np.zeros((n_samples, n_points))

    times = np.array(times)
    sorted_ind = times.argsort()

    for i in range(n_samples):
        sample_flux[i] = spectra[i][:, 0]
        sample_flux_var[i] = spectra[i][:, 1]

    # Make sure they are in time-wise order
    times = times[sorted_ind]
    sample_flux = sample_flux[sorted_ind]
    sample_flux_var = sample_flux_var[sorted_ind]

    if time_interp_method is None:
        time_interp_method = _default_interp_method

    if time_interp_method == 'linear':
        interp_flux, interp_flux_var = _linear_interp(times, sample_flux,
                                                      sample_flux_var,
                                                      interp_time)

    max_flux = interp_flux.max()
    interp_flux /= max_flux
    interp_flux_var /= max_flux**2

    return interp_flux, interp_flux_var
