import numpy as np
import matplotlib.pyplot as plt

from SNEx import SNEx


_data_dir = '../ex_data'

fn = f'{_data_dir}/ptf11kly_20110910_MJD55814.43.dat'

wave_range = None
model = SNEx(fn, wave_range=wave_range)

fit_range = (4000., 10000.)
# fit_range = None
x_pred = np.linspace(4100., 12500., 500)
y_nofil = model.predict(x_pred, regime='NIR', fit_range=fit_range,
                        extrap_method='planck', fit_method='ls',
                        bounds=([4000., 0.], [20000., np.inf]))
y_mono = model.predict(x_pred, regime='NIR', fit_range=fit_range,
                       extrap_method='planck', fit_method='ls',
                       filter_method='monotonic',
                       bounds=([4000., 0.], [30000., np.inf]))

fig, ax = plt.subplots()

ax.plot(model.data[:, 0], model.data[:, 1], 'k-', label='data')
ax.plot(x_pred, y_nofil, 'r-', label='no filter')
ax.plot(x_pred, y_mono, 'b-', label='monotonic filter')

ax.set_xlim(2000, 12500)
ax.set_ylim(0., 1.2e-12)

ax.legend()

fn = './example.pdf'
fig.savefig(fn)
