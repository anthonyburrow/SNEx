import numpy as np
import matplotlib.pyplot as plt

from SNEx import SNEx


_data_dir = '../ex_data'

fn = f'{_data_dir}/ptf11kly_20110910_MJD55814.43.dat'

wave_range = None
model = SNEx(fn, wave_range=wave_range)

# Check if < 5500 breaks this
fit_range = (5500., 7000.)

x_pred = np.linspace(4100., 12500., 500)

y_planck = model.predict(x_pred=x_pred, regime='NIR', fit_range=fit_range,
                         extrap_method='planck', fit_method='ls',
                         filter_method='monotonic',
                         bounds=([4000., 0.], [30000., np.inf]))

y_pca, x_pca = model.predict(regime='NIR', time=0., fit_range=fit_range,
                             extrap_method='pca', n_components=6)

fig, ax = plt.subplots()

ax.plot(model.data[:, 0], model.data[:, 1], 'k-', label='data')
ax.plot(x_pca, y_pca, 'r-', label='pca')
ax.plot(x_pred, y_planck, 'b-', label='planck')

ax.set_xlim(2000, 12500)
ax.set_ylim(0., 1.2e-12)

ax.axvline(fit_range[0], 0, 0.4)
ax.axvline(fit_range[1], 0, 0.4)

ax.legend()

fn = './example.pdf'
fig.savefig(fn)
