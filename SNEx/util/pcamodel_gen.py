import numpy as np
from pathlib import Path
import os

from .pcamodel import PCAModel
from .feature_ranges import feature_ranges


# Model properties
n_components = 15
predict_time_threshold = 2.
single_time_threshold = 1.

try:
    _spex_interp_dir = f'{Path.home()}/dev/SNEx_gen/model_scripts/time_interp/spex'
    if not os.path.isdir(_spex_interp_dir):
        raise FileNotFoundError
except FileNotFoundError:
    _spex_interp_dir = f'C:/dev/SNEx_gen/model_scripts/time_interp/spex'


# Defaults
_default_nir_predict = (5500., 8000.)
_default_uv_predict = (3800., 5500.)
_default_interp_method = 'linear'

# Same wave range from spextractor interpolation
total_wave_range = (3500., 10000.)
total_n_points = 2000

total_wave = np.linspace(*total_wave_range, total_n_points)


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


def _get_interp_spectrum(times, spectra, interp_time, time_interp_method=None):
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
        return _linear_interp(times, sample_flux, sample_flux_var, interp_time)


def _filter_spectrum(spec_file, interp_time, spec_time, wave_mask):
    # Only use spectra within predict_time_threshold
    if abs(spec_time - interp_time) > predict_time_threshold:
        return None

    flux = np.loadtxt(spec_file)

    flux = flux[wave_mask]
    if np.isnan(flux[:, 0]).any():
        return None

    max_flux = flux[:, 0].max()
    flux[:, 0] /= max_flux
    flux[:, 1] /= max_flux**2

    return flux


def _get_wave_mask(predict_range=None, predict_features=None, regime=None,
                   *args, **kwargs):
    '''Get overall mask for where we want to actually predict at.'''
    if predict_features is not None and predict_features:
        mask = np.full(total_n_points, False)
        for feature in predict_features:
            wave_range = feature_ranges[feature]
            sub_mask = \
                (wave_range[0] <= total_wave) & (total_wave <= wave_range[1])
            mask += sub_mask

        return mask

    if predict_range is None:
        if regime is None:
            regime = 'nir'
        regime = regime.lower()
        if regime == 'nir':
            predict_range = _default_nir_predict
        elif regime == 'uv':
            predict_range = _default_uv_predict

    mask = (predict_range[0] <= total_wave) & (total_wave <= predict_range[1])
    return mask


def _get_spectra(interp_time, wave_mask):
    training_flux = []
    training_flux_var = []

    for sn in os.listdir(_spex_interp_dir):
        spec_files = os.listdir(f'{_spex_interp_dir}/{sn}')
        spec_times = [float(fn[6:-4]) for fn in spec_files]
        spec_files = [f'{_spex_interp_dir}/{sn}/{f}' for f in spec_files]

        # Get valid spectra for each SN, then inter/extrapolate
        spectra = [_filter_spectrum(spec_file, interp_time, spec_time, wave_mask)
                   for spec_file, spec_time in zip(spec_files, spec_times)]
        spec_times = [t for t, x in zip(spec_times, spectra) if x is not None]
        spectra = [x for x in spectra if x is not None]

        n_valid = len(spectra)
        if n_valid == 0:
            continue
        elif n_valid == 1:
            print(f'Single spectrum valid for {sn}')
            if abs(spec_times[0] - interp_time) > single_time_threshold:
                continue
            flux = spectra[0][:, 0]
            flux_var = spectra[0][:, 1]
        elif n_valid > 1:
            print(f'Interpolating {sn}')
            flux, flux_var = _get_interp_spectrum(spec_times, spectra,
                                                  interp_time)

        training_flux.append(flux)
        training_flux_var.append(flux_var)

    return np.array(training_flux), np.array(training_flux_var)


def gen_model(time, *args, **kwargs):
    # Establish training data
    wave_mask = _get_wave_mask(*args, **kwargs)
    wave = total_wave[wave_mask]

    training_flux, training_flux_var = _get_spectra(time, wave_mask)

    # Create model and calculate eigenvectors
    model = PCAModel(wave, training_flux, training_flux_var,
                     n_components=n_components)
    model.calc_eig()

    return model
