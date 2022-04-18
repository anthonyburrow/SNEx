import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from SNEx import SNEx
from SNEx.util.feature_ranges import feature_ranges


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
    'predict_range': (5500., 8500.),
    # 'predict_features': ['Si II 6355'],
    'fit_features': ('Si II 5972', 'Si II 6355'),
    # 'fit_range': (5500., 7000.),
    'n_components': 10,
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

fig, ax = plt.subplots()

ax.plot(model.data[:, 0], model.data[:, 1], 'k-', label='data', zorder=6)
ax.fill_between(model.data[:, 0], model.data[:, 1] - model.data[:, 2],
                model.data[:, 1] + model.data[:, 2], color='#595959',
                zorder=4)

ax.plot(x_pca, y_pca, 'r-', label='pca', zorder=7)
ax.fill_between(x_pca, y_pca - y_err_pca, y_pca + y_err_pca, color='#ff7d7d',
                zorder=5)

# ax.plot(x_planck, y_planck, 'b-', label='planck')

for feature in params['fit_features']:
    wave_range = feature_ranges[feature]
    ax.axvspan(*wave_range, alpha=0.3)
    ax.axvline(wave_range[0], color='k', ls='--', lw=0.8, zorder=10)
    ax.axvline(wave_range[1], color='k', ls='--', lw=0.8, zorder=10)

ax.set_xlim(3500, 9200)
# ax.set_ylim(2.e-14, 1.2e-12)
ax.set_ylim(0., 1.2e-12)

ax.xaxis.set_minor_locator(MultipleLocator(250))

# ax.set_yscale('log')
ax.xaxis.grid(which='both', zorder=0)

ax.set_xlabel('Rest wavelength (A)')
ax.set_ylabel('Flux')

ax.legend()

plt.tight_layout()
fn = './example.pdf'
fig.savefig(fn)
