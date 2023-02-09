import numpy as np
from pathlib import Path
import os
import pickle

from .pcamodel import PCAModel
from ..util.feature_ranges import feature_ranges
from ..util.misc import between_mask, get_normalization


# Model properties
n_components = 10
allowed_extrapolation_time = 1.
allowed_no_extrapolation_time = 5.

try:
    _snexgen_dir = f'{Path.home()}/dev/SNEx_gen'
    if not os.path.isdir(_snexgen_dir):
        raise FileNotFoundError
except FileNotFoundError:
    _snexgen_dir = f'C:/dev/SNEx_gen'

_spex_interp_dir = f'{_snexgen_dir}/model_scripts/spex'
_csp_spex_dir = f'{_spex_interp_dir}/csp'
_nir_spex_dir = f'{_spex_interp_dir}/nir'

_interp_model_dir = f'{_snexgen_dir}/model_scripts/time_interp'
_csp_model_dir = f'{_interp_model_dir}/csp'
_nir_model_dir = f'{_interp_model_dir}/nir'

# Time range information
dtype = [('sn', 'U16'), ('t0', np.float64), ('t1', np.float64)]

fn = f'{_interp_model_dir}/time_info_csp.txt'
csp_time_info = np.loadtxt(fn, dtype=dtype)

fn = f'{_interp_model_dir}/time_info_nir.txt'
nir_time_info = np.loadtxt(fn, dtype=dtype)

# Defaults
_default_nir_predict = (5500., 8000.)
_default_uv_predict = (3800., 5500.)

# Get wave ranges from spextractor/time interpolation
csp_total_wave_range = (3500., 10000.)
csp_required_range = (4000., 8400.)
csp_total_n_points = 2000
csp_total_wave = np.linspace(*csp_total_wave_range, csp_total_n_points)
csp_cutoff_mask = between_mask(csp_total_wave, csp_required_range)
csp_total_wave = csp_total_wave[csp_cutoff_mask]

nir_total_wave_range = (7500., 24000.)
nir_required_range = (8400., 23950.)
nir_total_n_points = 1500
nir_total_wave = np.linspace(*nir_total_wave_range, nir_total_n_points)
nir_cutoff_mask = between_mask(nir_total_wave, nir_required_range)
nir_total_wave = nir_total_wave[nir_cutoff_mask]


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


def _get_fit_mask(wave, fit_features=None, fit_range=None, *args, **kwargs):
    fit_mask = np.full(len(wave), False)

    if fit_features is not None and fit_features:
        for feature in fit_features:
            wave_range = feature_ranges[feature]
            fit_mask += between_mask(wave, wave_range)

    if fit_range is not None:
        fit_mask += between_mask(wave, fit_range)

    return fit_mask


def _choose_spectrum_no_extrapolation(data_set, sn, predict_time, wave_mask):
    if data_set == 'csp':
        cutoff_mask = csp_cutoff_mask
        sn_dir = f'{_csp_spex_dir}/{sn}'
    elif data_set == 'nir':
        cutoff_mask = nir_cutoff_mask
        sn_dir = f'{_nir_spex_dir}/{sn}'

    try:
        spec_files = os.listdir(sn_dir)
    except FileNotFoundError:
        return None, None

    spec_files = [f for f in spec_files if f[-4:] == '.dat']
    spec_files = [f'{sn_dir}/{f}' for f in spec_files]

    # Get valid spectra for each SN, then inter/extrapolate
    spectra = []
    spectra_var = []
    spec_times = []
    for spec_file in spec_files:
        spec_time = float(spec_file[len(sn_dir) + 1 + 6:-4])
        # if abs(spec_time - predict_time) > allowed_no_extrapolation_time:
        #     continue
        if not -5 < spec_time - predict_time < 10:
            continue

        data = np.loadtxt(spec_file)
        data = data[cutoff_mask]
        data = data[wave_mask]

        flux = data[:, 0]
        var = data[:, 1]

        # Don't use spectrum if ANY of the wave points are missing
        if np.isnan(flux).any():
            continue

        spectra.append(flux)
        spectra_var.append(var)
        spec_times.append(spec_time)

    n_valid = len(spectra)

    if n_valid == 0:
        return None, None
    elif n_valid == 1:
        return_flux = spectra[0]
        return_var = spectra_var[0]
    elif n_valid > 1:
        closest_ind = abs(np.array(spec_times) - predict_time).argmin()
        return_flux = spectra[closest_ind]
        return_var = spectra_var[closest_ind]

    return return_flux, return_var


def _choose_spectrum_extrapolation(data_set, sn, predict_time, wave_mask):
    if data_set == 'csp':
        model_dir = _csp_model_dir
        time_info = csp_time_info
    elif data_set == 'nir':
        model_dir = _nir_model_dir
        time_info = nir_time_info

    # Load model if possible
    model_fn = f'{model_dir}/model_{sn}.pkl'
    with open(model_fn, 'rb') as file:
        model = pickle.load(file)

    # Check to see if model is in time range
    item = time_info[time_info['sn'] == sn]
    t0, t1 = item['t0'][0], item['t1'][0]

    t0 -= allowed_extrapolation_time
    t1 += allowed_extrapolation_time

    if not t0 < predict_time < t1:
        raise FileNotFoundError

    # Predict using the GPR
    flux, var = model.predict(np.array([[predict_time]]))
    flux = flux.squeeze()
    var = var.squeeze()

    var = var * np.ones(len(flux))

    flux = flux[wave_mask]
    var = var[wave_mask]

    # Check NaNs (problematic if this occurs...)
    if np.isnan(flux).any():
        raise FileNotFoundError(f'ERROR: NaNs for {sn}')

    return flux, var


def _choose_spectrum(*args, **kwargs):
    try:
        flux, var = _choose_spectrum_extrapolation(*args, **kwargs)
        extrapolated = True
    except FileNotFoundError:
        flux, var = _choose_spectrum_no_extrapolation(*args, **kwargs)
        extrapolated = False

    return flux, var, extrapolated


def _scale_nir(csp_flux, csp_wave_mask, nir_flux, nir_wave_mask):
    can_direct_scale = csp_wave_mask[-1] and nir_wave_mask[0]

    if can_direct_scale:
        # return csp_flux[-1] / nir_flux[0]
        return csp_flux[-2:].mean() / nir_flux[:2].mean()

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
        'bounds': ((10_000., -np.inf), (30_000., np.inf))
    }
    # Do a [0:2] slice just so x_pred will be an array; should be inconsequential
    planck_fit = planck_model.predict(x_pred=nir_total_wave[nir_wave_mask][0:2],
                                      **params)

    return planck_fit[0] / nir_flux[0]


def _get_spectra(predict_time, csp_wave_mask, nir_wave_mask):
    training_flux = []
    training_flux_var = []

    includes_opt = np.any(csp_wave_mask)   # PCA range covers optical region
    includes_nir = np.any(nir_wave_mask)   # PCA range covers NIR region

    count = 0
    n_csp_extrapolated = 0
    n_nir_extrapolated = 0
    for sn in os.listdir(_csp_spex_dir):
        if not includes_opt:
            continue

        csp_flux, csp_flux_var, csp_extrapolated = \
            _choose_spectrum('csp', sn, predict_time, csp_wave_mask)

        if csp_flux is None:
            continue

        if not includes_nir:
            if csp_extrapolated:
                n_csp_extrapolated += 1
            count += 1

            training_flux.append(csp_flux)
            training_flux_var.append(csp_flux_var)
            continue

        nir_flux, nir_flux_var, nir_extrapolated = \
            _choose_spectrum('nir', sn, predict_time, nir_wave_mask)

        if nir_flux is None:
            # No matching SN in the NIR
            continue

        # Put NIR data on the same scale as CSP
        # print(f'Scaling NIR data to CSP for {sn}...')
        nir_scale_factor = _scale_nir(csp_flux, csp_wave_mask,
                                      nir_flux, nir_wave_mask)
        nir_flux *= nir_scale_factor
        nir_flux_var *= nir_scale_factor**2

        # Compile CSP and NIR into one array
        flux = np.concatenate((csp_flux, nir_flux))
        flux_var = np.concatenate((csp_flux_var, nir_flux_var))

        # print(f'{count} : {sn} : csp {csp_time} : nir {nir_time}')
        training_flux.append(flux)
        training_flux_var.append(flux_var)

        if csp_extrapolated:
            n_csp_extrapolated += 1
        if nir_extrapolated:
            n_nir_extrapolated += 1
        count += 1

    for sn in os.listdir(_nir_spex_dir):
        if includes_opt:
            # optical is already accounted for
            continue

        nir_flux, nir_flux_var, nir_extrapolated = \
            _choose_spectrum('nir', sn, predict_time, nir_wave_mask)

        training_flux.append(nir_flux)
        training_flux_var.append(nir_flux_var)

        if nir_extrapolated:
            n_nir_extrapolated += 1
        count += 1

    msg = (f'Total sample size: {len(training_flux)}\n'
           f'Extrapolated: {n_csp_extrapolated} / {count} (CSP),'
           f' {n_nir_extrapolated} / {count} (NIR)')
    print(msg)

    return np.array(training_flux), np.array(training_flux_var)


def gen_model(time, *args, **kwargs):
    # Establish training data
    csp_wave_mask, nir_wave_mask = _get_wave_mask(*args, **kwargs)
    wave = np.concatenate((csp_total_wave[csp_wave_mask],
                           nir_total_wave[nir_wave_mask]))
    fit_mask = _get_fit_mask(wave, *args, **kwargs)

    training_flux, training_flux_var = \
        _get_spectra(time, csp_wave_mask, nir_wave_mask)

    # Normalize training data
    norm = get_normalization(training_flux[:, fit_mask], *args, **kwargs)
    training_flux = (training_flux.T / norm).T
    training_flux_var = (training_flux_var.T / norm**2).T

    # Create model and calculate eigenvectors
    model = PCAModel(wave, training_flux, training_flux_var,
                     n_components=n_components, *args, **kwargs)
    model.calc_eig()

    return model