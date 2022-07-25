import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from SNEx import SNEx
from SNEx.util.feature_ranges import feature_ranges
# from spextractor import Spextractor


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
    'time': 0.05,
    'extrap_method': 'pca',
    'fit_range': (5700., 8400.),
    'predict_range': (8400., 23000.),
    # 'fit_features': ('Si II 5972', 'Si II 6355'),
    # 'predict_features': ['O I', 'Ca II NIR'],
    'n_components': 8,
    'calc_var': True,
    'plot_pca': True,
    'norm_method': 'mean'
}
y_pca, y_err_pca, x_pca = model.predict(**params)

# Interpolation
# read_params['wave_range'] = (5500., 12000.)
# spex = Spextractor(fn, **read_params)

# y_interp, y_var_interp = spex.predict(x_pca)
# y_interp *= spex.fmax_out
# y_var_interp *= spex.fmax_out**2

# Plot
fig, ax = plt.subplots()

ax.set_xlim(0.3500, 2.3000)
# ax.set_ylim(7e-15, 1.2e-12)
# ax.set_ylim(0., 1.2e-12)

ax.plot(model.data[:, 0] / 1e4, model.data[:, 1], 'k-', label='data', zorder=6)
ax.fill_between(model.data[:, 0] / 1e4, model.data[:, 1] - model.data[:, 2],
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
    _x_pca = x_pca / 1e4
    _wave_range = (wave_range[0] / 1e4, wave_range[1] / 1e4)

    mask = (_wave_range[0] <= _x_pca) & (_x_pca <= _wave_range[1])

    x = _x_pca[mask]
    y = y_pca[mask]
    y_err = y_err_pca[mask]
    axis.plot(x, y, 'r-', zorder=7)
    axis.fill_between(x, y - y_err, y + y_err, color='#ff7d7d', zorder=5)

    axis.axvspan(*_wave_range, alpha=0.3, zorder=2)
    axis.axvline(_wave_range[0], color='k', ls='--', lw=0.8, zorder=10)
    axis.axvline(_wave_range[1], color='k', ls='--', lw=0.8, zorder=10)


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

ax.xaxis.set_minor_locator(MultipleLocator(0.0500))

ax.set_yscale('log')
ax.xaxis.grid(which='both', zorder=4)

ax.set_xlabel('Rest wavelength [$\mu m$]')
ax.set_ylabel('Flux')

ax.axvspan(1.2963, 1.4419, alpha=0.8, color='grey', zorder=3)
ax.axvspan(1.7421, 1.9322, alpha=0.8, color='grey', zorder=3)

line = ax.plot([0.], [0.], 'r-', label='PCA reconstruction')
ax.legend()

plt.tight_layout()
fn = './example.pdf'
fig.savefig(fn)
fn = './example.png'
fig.savefig(fn, dpi=200)


# Plot
fig, ax = plt.subplots()

ax.set_xlim(0.3500, 0.9000)
ax.set_ylim(1.e-14, 1.3e-12)

ax.plot(model.data[:, 0], model.data[:, 1], 'k-', label='data', zorder=6)
ax.fill_between(model.data[:, 0], model.data[:, 1] - model.data[:, 2],
                model.data[:, 1] + model.data[:, 2], color='#595959',
                zorder=4)

ax.axvline(params['fit_range'][0], color='r', ls='--', lw=1.2, zorder=10)
ax.axvline(params['fit_range'][1], color='r', ls='--', lw=1.2, zorder=10)

ax.xaxis.set_minor_locator(MultipleLocator(0.0250))

ax.set_yscale('log')
ax.xaxis.grid(which='both', zorder=0)

ax.set_xlabel('Rest wavelength [$\mu m$]')
ax.set_ylabel('Flux')

plt.tight_layout()
fn = './example_no_pred.png'
fig.savefig(fn, dpi=200)


# Testing
# chisq = ((y_interp - y_pca)**2 / y_err_pca**2).sum() / (len(x_pca) - 1)
# print(chisq)
