import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from SNEx import SNEx
from SNEx.util.feature_ranges import feature_ranges
from spextractor import Spextractor


_data_dir = '../ex_data'

fn = f'{_data_dir}/ptf11kly_20110910_MJD55814.43.dat'

read_params = {
    'z': 0.001208,
    'wave_range': None,
    'host_EBV': 0.014,
    'host_RV': 1.4,
    'MW_EBV': 0.011
}
model = SNEx(fn, **read_params)

params = {
    'regime': 'nir',
    'time': 0.,
    'extrap_method': 'pca',
    'fit_range': (5500., 7000.),
    'predict_range': (8701., 12000.),
    # 'fit_features': ('Si II 5972', 'Si II 6355'),
    # 'predict_features': ['O I', 'Ca II NIR'],
    'n_components': 6,
    'calc_var': True,
    'plot_pca': True
}
y_pca, y_err_pca, x_pca = model.predict(**params)

# x_planck = np.linspace(4000., 9000.)
# params = {
#     'regime': 'NIR',
#     'time': 0.,
#     'fit_range': (4000., 6500.),
#     'extrap_method': 'planck',
#     'filter_method': 'monotonic'
# }
# y_planck = model.predict(x_pred=x_planck, **params)

# Interpolation
read_params['wave_range'] = (5500., 12000.)
spex = Spextractor(fn, **read_params)

y_interp, y_var_interp = spex.predict(x_pca)
y_interp *= spex.fmax_out
y_var_interp *= spex.fmax_out**2

mask = x_pca > params['predict_range'][0]
offset = (y_pca[mask] / y_interp[mask])
print(f'Offset: {offset.mean()} +- {offset.std()}')

const_offset = offset.mean()
y_pca[mask] /= const_offset
y_err_pca[mask] /= const_offset

# Plot
fig, ax = plt.subplots()

ax.set_xlim(3500., 12000.)
# ax.set_ylim(2.e-14, 1.2e-12)
# ax.set_ylim(0., 1.2e-12)

ax.plot(model.data[:, 0], model.data[:, 1], 'k-', label='data', zorder=6)
ax.fill_between(model.data[:, 0], model.data[:, 1] - model.data[:, 2],
                model.data[:, 1] + model.data[:, 2], color='#595959',
                zorder=4)


def plot_feature(axis, feature):
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


def plot_range(axis, wave_range):
    mask = (wave_range[0] <= x_pca) & (x_pca <= wave_range[1])

    x = x_pca[mask]
    y = y_pca[mask]
    y_err = y_err_pca[mask]
    axis.plot(x, y, 'r-', zorder=7)
    axis.fill_between(x, y - y_err, y + y_err, color='#ff7d7d', zorder=5)

    axis.axvspan(*wave_range, alpha=0.3)
    axis.axvline(wave_range[0], color='k', ls='--', lw=0.8, zorder=10)
    axis.axvline(wave_range[1], color='k', ls='--', lw=0.8, zorder=10)


try:
    for feature in params['fit_features']:
        plot_feature(ax, feature)
except KeyError:
    plot_range(ax, params['fit_range'])

try:
    for feature in params['predict_features']:
        plot_feature(ax, feature)
except KeyError:
    plot_range(ax, params['predict_range'])

ax.xaxis.set_minor_locator(MultipleLocator(250))

ax.set_yscale('log')
ax.xaxis.grid(which='both', zorder=0)

ax.set_xlabel('Rest wavelength (A)')
ax.set_ylabel('Flux')

line = ax.plot([0.], [0.], 'r-', label='PCA')
ax.legend()

plt.tight_layout()
fn = './example.pdf'
fig.savefig(fn)


# Testing
# chisq = ((y_interp - y_pca)**2 / y_err_pca**2).sum() / (len(x_pca) - 1)
# print(chisq)
