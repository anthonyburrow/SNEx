import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import pandas as pd
from pandas import ExcelFile, read_excel

from SNEx import SNEx
from SNEx.util.feature_ranges import feature_ranges
from spextractor import Spextractor

_home_dir = str(Path.home())
_data_dir = f'{_home_dir}/data'

fn = f'{_data_dir}/CSP/CSP.xlsx'
xls = ExcelFile(fn)
df_I = read_excel(xls, 'CSPI')
df_II = read_excel(xls, 'CSPII')

fn = f'{_data_dir}/FIRE/Packed_FIRE_spec_SNIa.pkl'
df_nir = pd.read_pickle(fn)

spec_files = {
    'SN2014D': 'CSP14aaa_NOT_ALFOSC_gr4_11jan2014.fits',
    'SN2012ij': 'SN12ij_b01_DUP_WF_10jan13.fits',
    'ASAS15bm': 'ASAS15bm_b01_DUP_WF_02feb15.fits'
}

nir_times = {
    'SN2014D': '1.100',
    'SN2012ij': '-4.181',
    'ASAS15bm': '3.156'
}

# SN2014D (csp 3.875, nir 1.100) : Core-normal (93.91%)
# SN2012ij (csp -0.178, nir -4.181) : Cool (97.40%)
# ASAS15bm (csp 1.171, nir 3.156) : Shallow-silicon (97.74%)

prediction_params = {
    'regime': 'nir',
    'time': 0.,
    'extrap_method': 'pca',
    'fit_range': (5500., 8400.),
    'predict_range': (8400., 23000.),
    # 'fit_features': ('Si II 5972', 'Si II 6355'),
    # 'predict_features': ['O I', 'Ca II NIR'],
    'n_components': 8,
    'calc_var': True,
    'plot_pca': False
}


def get_read_params(name):
    sn = df_I.loc[df_I['Name'] == name]
    if sn.empty:
        sn = df_II.loc[df_II['Name'] == name]
        z = sn['zhel'].item()
    else:
        z = sn['zmcb'].item()

    MW_EBV = sn['E(B-V) MW'].item()
    host_EBV = sn['E(B-V)'].item()
    host_RV = sn['R_V Host'].item()

    read_params = {
        'z': z,
        'wave_range': None,
        'host_EBV': host_EBV,
        'host_RV': host_RV,
        'MW_EBV': MW_EBV
    }
    return read_params


def plot_feature(axis, feature, x_pca, y_pca, y_err_pca):
    wave_range = feature_ranges[feature]
    mask = (wave_range[0] <= x_pca) & (x_pca <= wave_range[1])

    x = x_pca[mask]
    y = y_pca[mask]
    y_err = y_err_pca[mask]
    axis.plot(x, y, 'r-', zorder=7)
    axis.fill_between(x, y - y_err, y + y_err, color='#ff7d7d', zorder=5)

    axis.axvspan(*wave_range, alpha=0.3)
    axis.axvline(wave_range[0], color='k', ls='--', lw=0.8, zorder=10)
    axis.axvline(wave_range[1], color='k', ls='--', lw=0.8, zorder=10)

    x_pos, y_pos = wave_range[0] + 30., 0.018 * axis.get_ylim()[1]
    axis.text(x_pos, y_pos, feature, rotation=90., fontsize=8.)


def plot_range(axis, wave_range, x_pca, y_pca, y_err_pca):
    _x_pca = x_pca / 1e4
    _wave_range = (wave_range[0] / 1e4, wave_range[1] / 1e4)

    mask = (_wave_range[0] <= _x_pca) & (_x_pca <= _wave_range[1])

    x = _x_pca[mask]
    y = y_pca[mask]
    y_err = y_err_pca[mask]
    axis.plot(x, y, 'r-', zorder=7)
    axis.fill_between(x, y - y_err, y + y_err, color='#ff7d7d', zorder=5)

    axis.axvspan(*_wave_range, alpha=0.3)
    axis.axvline(_wave_range[0], color='k', ls='--', lw=0.8, zorder=10)
    axis.axvline(_wave_range[1], color='k', ls='--', lw=0.8, zorder=10)


def concat_spectra(name, cutoff=8500.):
    # CSP
    spec_file = f'{_data_dir}/CSP_I_II_spectra/{name}/{spec_files[name]}'
    read_params = get_read_params(name)
    read_params['wave_range'] = (3500., cutoff)
    spex = Spextractor(spec_file, **read_params)
    csp_data = spex.data

    # NIR
    for i, sn in df_nir.iterrows():
        if name != sn['SN']:
            continue

        tmax = sn['Tmax (MJD)']
        time = (sn['JD'] - 2400000.5) - tmax
        time = f'{time:.3f}'

        if time != nir_times[name]:
            continue

        mask = sn['wave_rest'] > (cutoff / 1.e4)
        wave_rest = sn['wave_rest'][mask] * 1.e4
        flux = sn['flux'][mask]
        flux_err = sn['flux_err'][mask]

        nir_data = np.zeros((len(wave_rest), 3))
        nir_data[:, 0] = wave_rest
        nir_data[:, 1] = flux
        nir_data[:, 2] = flux_err

        break

    scale = csp_data[-1, 1] / nir_data[0, 1]
    nir_data[:, 1:3] *= scale

    return np.concatenate((csp_data, nir_data), axis=0)


def full_prediction(name):
    obs_data = concat_spectra(name)

    model = SNEx(obs_data)
    y_pca, y_err_pca, x_pca = model.predict(**prediction_params)

    # Plot
    fig, ax = plt.subplots()

    ax.set_xlim(0.3500, 2.3000)
    # ax.set_ylim(2.e-14, 1.2e-12)
    # ax.set_ylim(0., 1.2e-12)

    ax.plot(obs_data[:, 0] / 1e4, obs_data[:, 1], 'k-', label='data', zorder=6)
    ax.fill_between(obs_data[:, 0] / 1e4, obs_data[:, 1] - obs_data[:, 2],
                    obs_data[:, 1] + obs_data[:, 2], color='#595959', zorder=4)

    try:
        for feature in prediction_params['fit_features']:
            plot_feature(ax, feature, x_pca, y_pca, y_err_pca)
    except KeyError:
        plot_range(ax, prediction_params['fit_range'], x_pca, y_pca, y_err_pca)

    try:
        for feature in prediction_params['predict_features']:
            plot_feature(ax, feature, x_pca, y_pca, y_err_pca)
    except KeyError:
        plot_range(ax, prediction_params['predict_range'], x_pca, y_pca, y_err_pca)

    ax.xaxis.set_minor_locator(MultipleLocator(0.0500))

    ax.set_yscale('log')
    ax.xaxis.grid(which='both', zorder=0)

    ax.set_xlabel('Rest wavelength [$\mu m$]')
    ax.set_ylabel('Flux')

    ax.axvspan(1.2963, 1.4419, alpha=0.8, color='grey', zorder=3)
    ax.axvspan(1.7421, 1.9322, alpha=0.8, color='grey', zorder=3)

    ax.plot([0.], [0.], 'r-', label='PCA reconstruction')
    ax.legend()

    plt.tight_layout()
    fn = f'./example_{name}.pdf'
    fig.savefig(fn)
    fn = f'./example_{name}.png'
    fig.savefig(fn, dpi=200)


for sn in spec_files:
    full_prediction(sn)
