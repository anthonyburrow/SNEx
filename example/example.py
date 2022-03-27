import numpy as np
import matplotlib.pyplot as plt

from SNEx import SNEx


_data_dir = '../ex_data'

fn = f'{_data_dir}/ptf11kly_20110910_MJD55814.43.dat'

wave_range = None
model = SNEx(fn, wave_range=wave_range)

# Check if < 5500 breaks this
fit_range = (5500., 6500.)

x_pred = np.linspace(4100., 12500., 500)

y_pca, y_err_pca, x_pca = model.predict(regime='NIR', time=0.,
                                        fit_range=fit_range,
                                        extrap_method='pca', n_components=8,
                                        calc_var=True)

fig, ax = plt.subplots()

ax.plot(model.data[:, 0], model.data[:, 1], 'k-', label='data')
ax.fill_between(model.data[:, 0], model.data[:, 1] - model.data[:, 2],
                model.data[:, 1] + model.data[:, 2], color='#595959')

ax.plot(x_pca, y_pca, 'r-', label='pca')
ax.fill_between(x_pca, y_pca - y_err_pca, y_pca + y_err_pca, color='#ff7d7d')

# ax.plot(x_pred, y_planck, 'b-', label='planck')

ax.set_xlim(3000, 9000)
ax.set_ylim(1.e-14, 1.2e-12)

# ax.axvline(fit_range[0], 0, 0.6)
ax.axvline(fit_range[1], 0, 0.6)

ax.set_yscale('log')

ax.legend()

fn = './example.pdf'
fig.savefig(fn)
