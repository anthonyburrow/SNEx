import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from SNEx import SNEx
from SNEx.util.feature_ranges import feature_ranges


fn = f'./ptf11kly_20110910_MJD55814.43.dat'

read_params = {
    'z': 0.001208,
    'wave_range': None,
    'host_EBV': 0.014,
    'host_RV': 1.4,
    'MW_EBV': 0.011
}
model = SNEx(fn, **read_params)

params = {
    'time': 0.05,
    'extrap_method': 'pca',
    'fit_range': (5000., 8400.),
    'predict_range': (8400., 23000.),
    'n_components': 10,
    'calc_var': True,
    'plot_pca': True,
}
y_pca, y_err_pca, x_pca = model.predict(**params)

# Plot
fig, ax = plt.subplots()

ax.set_xlim(0.3500, 2.3000)
# ax.set_ylim(7e-15, 1.2e-12)
# ax.set_ylim(0., 1.2e-12)

ax.plot(model.data[:, 0] * 1e-4, model.data[:, 1], 'k-', label='Observed',
        zorder=6)
ax.fill_between(model.data[:, 0] * 1e-4, model.data[:, 1] - model.data[:, 2],
                model.data[:, 1] + model.data[:, 2], color='#595959',
                zorder=4)

ax.plot(x_pca * 1e-4, y_pca, 'r-', zorder=7)
ax.fill_between(x_pca * 1e-4, y_pca - y_err_pca, y_pca + y_err_pca,
                color='#ff7d7d', zorder=5)

# Show fit range
ax.axvline(params['fit_range'][0] * 1e-4, color='r', ls='--', lw=0.8, zorder=10)
ax.axvline(params['fit_range'][1] * 1e-4, color='r', ls='--', lw=0.8, zorder=10)

# Tellurics
ax.axvspan(1.2963, 1.4419, alpha=0.8, color='grey', zorder=3)
ax.axvspan(1.7421, 1.9322, alpha=0.8, color='grey', zorder=3)

ax.xaxis.set_minor_locator(MultipleLocator(0.0500))

ax.set_yscale('log')
ax.xaxis.grid(which='both', zorder=4)

ax.set_xlabel('Rest wavelength [$\mu m$]')
ax.set_ylabel('Flux')

line = ax.plot([0.], [0.], 'r-', label='PCA reconstruction')
ax.legend()

fn = './11fe_single.png'
fig.savefig(fn, dpi=200)
