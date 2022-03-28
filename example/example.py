import numpy as np
import matplotlib.pyplot as plt

from SNEx import SNEx


_data_dir = '../ex_data'

fn = f'{_data_dir}/ptf11kly_20110910_MJD55814.43.dat'
z = 0.001208

wave_range = None
model = SNEx(fn, z=z, wave_range=wave_range)

params = {
    'regime': 'NIR',
    'time': 0.,
    'fit_range': (5000., 6500.),
    'extrap_method': 'pca',
    'n_components': 10,
    'calc_var': True
}
y_pca, y_err_pca, x_pca = model.predict(**params)

fig, ax = plt.subplots()

ax.plot(model.data[:, 0], model.data[:, 1], 'k-', label='data')
ax.fill_between(model.data[:, 0], model.data[:, 1] - model.data[:, 2],
                model.data[:, 1] + model.data[:, 2], color='#595959')

ax.plot(x_pca, y_pca, 'r-', label='pca')
ax.fill_between(x_pca, y_pca - y_err_pca, y_pca + y_err_pca, color='#ff7d7d')

for i in range(2):
    wave = params['fit_range'][i]
    endpoint_ind = abs(model.data[:, 0] - wave).argmin()
    flux = model.data[:, 1][endpoint_ind]
    ax.vlines(wave, 0., flux, linestyles='dashed')

ax.set_xlim(3500, 9200)
ax.set_ylim(1.e-14, 1.2e-12)

# ax.set_yscale('log')

ax.legend()

fn = './example.pdf'
fig.savefig(fn)
