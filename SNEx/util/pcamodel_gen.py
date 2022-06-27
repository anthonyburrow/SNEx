import numpy as np
from pathlib import Path
import os

from .pcamodel import PCAModel
from .pcaplot import plot_training
from .feature_ranges import feature_ranges
from .misc import between_mask

# import matplotlib.pyplot as plt

# Model properties
n_components = 8
time_threshold_0 = 5.
time_threshold_30 = 10.
time_threshold_min = 2.

try:
    _snexgen_dir = f'{Path.home()}/dev/SNEx_gen'
    if not os.path.isdir(_snexgen_dir):
        raise FileNotFoundError
except FileNotFoundError:
    _snexgen_dir = f'C:/dev/SNEx_gen'

_spex_interp_dir = f'{_snexgen_dir}/model_scripts/time_interp/spex'

# Defaults
_default_nir_predict = (5500., 8000.)
_default_uv_predict = (3800., 5500.)

# Get wave ranges from spextractor interpolation
csp_total_wave_range = (3500., 10000.)
csp_total_n_points = 2000

nir_total_wave_range = (7500., 24000.)
nir_total_n_points = 1500

# Account for overlap, setup complete wavelength array
csp_nir_cutoff = 8500.

csp_total_wave = np.linspace(*csp_total_wave_range, csp_total_n_points)
nir_total_wave = np.linspace(*nir_total_wave_range, nir_total_n_points)

csp_cutoff_mask = csp_total_wave <= csp_nir_cutoff
nir_cutoff_mask = nir_total_wave > csp_nir_cutoff

csp_total_wave = csp_total_wave[csp_cutoff_mask]
nir_total_wave = nir_total_wave[nir_cutoff_mask]


def _calc_time_window(interp_time):
    slope = (time_threshold_30 - time_threshold_0) / 30.
    window = slope * interp_time + time_threshold_0

    if window < time_threshold_min:
        return time_threshold_min

    return window


def _get_wave_mask(regime=None, predict_range=None, predict_features=None,
                   fit_features=None, fit_range=None, *args, **kwargs):
    '''Get overall mask for where we want to actually predict at.'''
    csp_mask = np.full(len(csp_total_wave), False)
    nir_mask = np.full(len(nir_total_wave), False)

    if predict_features is not None and predict_features:
        for feature in predict_features:
            wave_range = feature_ranges[feature]
            csp_mask += between_mask(csp_total_wave, wave_range)
            nir_mask += between_mask(nir_total_wave, wave_range)
    elif predict_range is None:
        if regime is None:
            regime = 'nir'
        regime = regime.lower()
        if regime == 'nir':
            predict_range = _default_nir_predict
        elif regime == 'uv':
            predict_range = _default_uv_predict

    if predict_range is not None:
        csp_mask += between_mask(csp_total_wave, predict_range)
        nir_mask += between_mask(nir_total_wave, predict_range)

    if fit_features is not None and fit_features:
        for feature in fit_features:
            wave_range = feature_ranges[feature]
            csp_mask += between_mask(csp_total_wave, wave_range)
            nir_mask += between_mask(nir_total_wave, wave_range)

    if fit_range is not None:
        csp_mask += between_mask(csp_total_wave, fit_range)
        nir_mask += between_mask(nir_total_wave, fit_range)

    return csp_mask, nir_mask


def _choose_spectrum(data_set, sn, predict_time, wave_mask):
    if data_set == 'csp':
        cutoff_mask = csp_cutoff_mask
        sn_dir = f'{_spex_interp_dir}/csp/{sn}'
    elif data_set == 'nir':
        cutoff_mask = nir_cutoff_mask
        sn_dir = f'{_spex_interp_dir}/nir/{sn}'

    try:
        spec_files = os.listdir(sn_dir)
    except FileNotFoundError:
        return None, None
    spec_files = [f for f in spec_files if f[-4:] == '.dat']
    spec_files = [f'{sn_dir}/{f}' for f in spec_files]

    # Get valid spectra for each SN, then inter/extrapolate
    spectra = []
    spec_times = []
    for spec_file in spec_files:
        spec_time = float(spec_file[len(sn_dir) + 1 + 6:-4])
        if abs(spec_time - predict_time) > _calc_time_window(predict_time):
            continue

        flux = np.loadtxt(spec_file)
        flux = flux[cutoff_mask]
        flux = flux[wave_mask]

        # Don't use spectrum if ANY of the wave points are missing
        if np.isnan(flux[:, 0]).any():
            continue

        # (This should move to the Spextractor interpolation process, pre-save)
        max_flux = flux[:, 0].max()
        flux[:, 0] /= max_flux
        flux[:, 1] /= max_flux**2

        spectra.append(flux)
        spec_times.append(spec_time)

    n_valid = len(spectra)

    if n_valid == 0:
        return None, None
    elif n_valid == 1:
        spectrum = spectra[0]
        time = spec_times[0]
    elif n_valid > 1:
        closest_ind = abs(np.array(spec_times) - predict_time).argmin()
        spectrum = spectra[closest_ind]
        time = spec_times[closest_ind]

    return spectrum, time


def _scale_nir(csp_flux, csp_wave_mask, nir_flux, nir_wave_mask):
    can_direct_scale = csp_wave_mask[-1] and nir_wave_mask[0]

    if can_direct_scale:
        return csp_flux[-1] / nir_flux[0]

    # Scale using a Planck function
    from ..SNEx import SNEx

    csp_data = np.zeros((len(csp_flux), 2))
    csp_data[:, 0] = csp_total_wave[csp_wave_mask]
    csp_data[:, 1] = csp_flux

    planck_model = SNEx(csp_data)
    params = {
        'regime': 'NIR',
        'extrap_method': 'planck',
        'filter_method': 'monotonic',
        'bounds': ((10000., -np.inf), (30000., np.inf))
    }
    # Do a [0:2] slice just so x_pred will be an array; should be inconsequential
    planck_fit = planck_model.predict(x_pred=nir_total_wave[nir_wave_mask][0:2],
                                      **params)

    return planck_fit[0] / nir_flux[0]


def _get_spectra(predict_time, csp_wave_mask, nir_wave_mask):
    training_flux = []
    training_flux_var = []

    csp_dir = f'{_spex_interp_dir}/csp'
    nir_dir = f'{_spex_interp_dir}/nir'

    is_csp = np.any(csp_wave_mask)
    is_nir = np.any(nir_wave_mask)

    for sn in os.listdir(csp_dir):
        if not is_csp:
            continue

        csp_spectrum, csp_time = _choose_spectrum('csp', sn, predict_time, csp_wave_mask)
        if csp_spectrum is None:
            continue
        csp_flux = csp_spectrum[:, 0]
        csp_flux_var = csp_spectrum[:, 1]

        if not is_nir:
            training_flux.append(csp_flux)
            training_flux_var.append(csp_flux_var)
            continue

        nir_spectrum, nir_time = _choose_spectrum('nir', sn, predict_time, nir_wave_mask)
        if nir_spectrum is None:
            continue
        nir_flux = nir_spectrum[:, 0]
        nir_flux_var = nir_spectrum[:, 1]

        # Put NIR data on the same scale as CSP
        # print(f'Scaling NIR data to CSP for {sn}...')
        nir_scale_factor = _scale_nir(csp_flux, csp_wave_mask,
                                      nir_flux, nir_wave_mask)
        nir_flux *= nir_scale_factor
        nir_flux_var *= nir_scale_factor**2

        # Compile CSP and NIR
        flux = np.concatenate((csp_flux, nir_flux))
        flux_var = np.concatenate((csp_flux_var, nir_flux_var))

        training_flux.append(flux)
        training_flux_var.append(flux_var)

    for sn in os.listdir(nir_dir):
        if is_csp:
            continue

        nir_spectrum = _choose_spectrum('nir', sn, predict_time, nir_wave_mask)
        if nir_spectrum is None:
            continue
        nir_flux = nir_spectrum[:, 0]
        nir_flux_var = nir_spectrum[:, 1]

        training_flux.append(nir_flux)
        training_flux_var.append(nir_flux_var)

    msg = (
        f'Total sample size within {_calc_time_window(predict_time)} days: '
        f'{len(training_flux)}\n'
    )
    print(msg)

    return np.array(training_flux), np.array(training_flux_var)


def gen_model(time, *args, **kwargs):
    # Establish training data
    csp_wave_mask, nir_wave_mask = _get_wave_mask(*args, **kwargs)
    wave = np.concatenate((csp_total_wave[csp_wave_mask],
                           nir_total_wave[nir_wave_mask]))

    training_flux, training_flux_var = _get_spectra(time, csp_wave_mask,
                                                    nir_wave_mask)

    plot_training(wave, training_flux, training_flux_var)

    # Create model and calculate eigenvectors
    model = PCAModel(wave, training_flux, training_flux_var,
                     n_components=n_components)
    model.calc_eig()

    return model
