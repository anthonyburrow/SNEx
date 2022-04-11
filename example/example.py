import numpy as np
import matplotlib.pyplot as plt

from SNEx import SNEx


_data_dir = '../ex_data'

fn = f'{_data_dir}/ptf11kly_20110910_MJD55814.43.dat'
z = 0.001208
host_EBV = 0.014
host_RV = 1.4
MW_EBV = 0.011

wave_range = None
model = SNEx(fn, z=z, wave_range=wave_range)

params = {
    'regime': 'nir',
    'time': 0.,
    'extrap_method': 'pca',
    'predict_range': (5500., 8000.),
    # 'predict_features': ['Si II 6355'],
    'fit_features': ('Si II 6355', ),
    # 'fit_range': (5500., 7000.),
    'n_components': 10,
    'calc_var': True
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

ax.plot(model.data[:, 0], model.data[:, 1], 'k-', label='data')
ax.fill_between(model.data[:, 0], model.data[:, 1] - model.data[:, 2],
                model.data[:, 1] + model.data[:, 2], color='#595959')

ax.plot(x_pca, y_pca, 'r-', label='pca')
ax.fill_between(x_pca, y_pca - y_err_pca, y_pca + y_err_pca, color='#ff7d7d')

# ax.plot(x_planck, y_planck, 'b-', label='planck')

# for i in range(2):
#     wave = params['fit_range'][i]
#     endpoint_ind = abs(model.data[:, 0] - wave).argmin()
#     flux = model.data[:, 1][endpoint_ind]
#     ax.vlines(wave, 0., flux, linestyles='dashed')

ax.set_xlim(3500, 9200)
ax.set_ylim(2.e-14, 1.2e-12)

# ax.set_yscale('log')

ax.legend()

fn = './example.pdf'
fig.savefig(fn)
