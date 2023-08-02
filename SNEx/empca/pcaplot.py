import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter, NullFormatter

from ..util.plot_setup import paper_plot
paper_plot()


def plot_info(pcamodel):
    plot_eigenvectors(pcamodel)
    plot_explained_var(pcamodel)
    plot_explained_var_cumsum(pcamodel)
    plot_training(pcamodel)
    plt.close('all')


highlight_regions = {
    'Si II': {
        'color': 'tab:blue',
        'regions': [
            (0.5650, 0.5900),
            (0.5950, 0.6355),
        ],
    },
    'O I': {
        'color': 'tab:brown',
        'regions': [
            (0.7, 0.77),
        ],
    },
    'Ca II': {
        'color': 'tab:green',
        'regions': [
            (0.78, 0.86),
        ],
    },
    'Mg II': {
        'color': 'tab:orange',
        'regions': [
            (0.87, 0.925),
            (0.95, 1.01),
            (1.02, 1.075),
        ],
    },
    'Si III': {
        'color': 'tab:red',
        'regions': [
            (1.15, 1.24),
            (1.26, 1.35),
        ],
    },
    'Fe/Co': {
        'color': 'tab:purple',
        'regions': [
            (1.5, 1.65),
        ],
    },
}


def plot_eigenvectors(pcamodel):
    n_comp = 4
    fig, ax = plt.subplots(n_comp, 1, sharey=True, sharex=True, figsize=(6, 6))

    wave = pcamodel.wave * 1e-4
    eig = pcamodel.eigenvectors

    for i, _ax in enumerate(ax):
        plot_params = {
            'ls': '-',
            'lw': 1.,
            'c': 'k'
        }
        _ax.semilogx(wave, eig[i], label=f'PC{i + 1}', **plot_params)
        _ax.axhline(0., color='grey', ls='--', zorder=-4)
        _ax.text(0.9, 0.8, f'PC$_{i + 1}$', transform=_ax.transAxes, fontsize=12)

        # Highlight regions
        plot_params['lw'] = 1.3
        for ion in highlight_regions:
            regions = highlight_regions[ion]['regions']
            for reg_min, reg_max in regions:
                mask = (reg_min < wave) & (wave < reg_max)
                plot_params['c'] = highlight_regions[ion]['color']
                _ax.semilogx(wave[mask], eig[i][mask], **plot_params)

        # _ax.axvline(0.9227, color='red', ls='--')
        # _ax.axvline(1.0092, color='red', ls='--')
        # _ax.axvline(1.0927, color='red', ls='--')

        # Telluric regions
        _ax.axvspan(1.2963, 1.4419, alpha=0.8, color='grey', zorder=-10)
        _ax.axvspan(1.7421, 1.9322, alpha=0.8, color='grey', zorder=-10)

    for ion in highlight_regions:
        regions = highlight_regions[ion]['regions']
        color = highlight_regions[ion]['color']

        x_pos = 0.5 * (regions[0][0] + regions[-1][-1])
        arrowprops = {
            'arrowstyle': '-[',
            'lw': 0.8,
            'shrinkA': 0.05,
            'color': color
        }
        ax[0].annotate(ion, xy=(x_pos, 0.03), xycoords='data',
                       xytext=(x_pos, 0.05), textcoords='data',
                       fontsize=8., arrowprops=arrowprops, ha='center',
                       va='center', c=color)

    ax[-1].set_xlabel(r'Rest wavelength ($\mu m$)', fontsize=12)
    fig.supylabel(r'$(F - \mu_F) / \sigma_F$', fontsize=12)

    ax[0].xaxis.set_minor_locator(MultipleLocator(0.05))
    ax[0].xaxis.set_major_locator(MultipleLocator(0.2))
    ax[0].xaxis.set_major_formatter(ScalarFormatter())
    ax[0].xaxis.set_minor_formatter(NullFormatter())

    y_interval = 0.05
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.01))
    ax[0].yaxis.set_major_locator(MultipleLocator(y_interval))
    plt.yticks([-y_interval, 0., y_interval])

    [_ax.tick_params(axis='both', which='major', labelsize=10, length=3.)
     for _ax in ax]
    [_ax.tick_params(axis='both', which='minor', length=1.8)
     for _ax in ax]

    # [_ax.grid(axis='x', which='both')
    #  for _ax in ax]

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.)

    plot_dir = './pca_info'
    fn = f'{plot_dir}/eigenvectors.pdf'
    fig.savefig(fn)
    fn = f'{plot_dir}/eigenvectors.png'
    fig.savefig(fn, dpi=200)

    plt.close('all')


def plot_explained_var(pcamodel):
    fig, ax = plt.subplots()

    pc_ids = np.arange(1, pcamodel.n_components + 1)
    ax.plot(pc_ids, pcamodel.explained_var, 'ko', ms=6.)

    ax.set_yscale('log')
    ax.set_ylim(top=1.)

    ax.set_xlabel('PC')
    ax.set_ylabel('Fractional explained variance')

    ax.xaxis.set_major_locator(MultipleLocator(1))

    plt.tight_layout()

    plot_dir = './pca_info'
    fn = f'{plot_dir}/explained_variance.pdf'
    fig.savefig(fn)
    fn = f'{plot_dir}/explained_variance.png'
    fig.savefig(fn, dpi=200)


def plot_explained_var_cumsum(pcamodel):
    fig, ax = plt.subplots()

    pc_ids = np.arange(1, pcamodel.n_components + 1)
    cumul_var = pcamodel.explained_var.cumsum()
    print('Cumulative var explained: ', cumul_var)
    ax.plot(pc_ids, cumul_var, 'ko', ms=6.)

    ax.axhline(0.95, c='r', ls='--')

    ax.set_ylim(0.4, 1.)

    ax.set_xlabel('PC')
    ax.set_ylabel('Cumulative Variance Explained')

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.tight_layout()

    plot_dir = './pca_info'
    fn = f'{plot_dir}/explained_variance_cumsum.pdf'
    fig.savefig(fn)
    fn = f'{plot_dir}/explained_variance_cumsum.png'
    fig.savefig(fn, dpi=200)


def plot_training(pcamodel):
    plot_dir = './training_spec'

    wave = pcamodel.wave
    training_flux = pcamodel.flux_train
    training_flux_var = pcamodel.flux_var_train

    training_flux, training_flux_var = pcamodel.descale(training_flux,
                                                        training_flux_var)

    fig, ax = plt.subplots()
    fig_compiled, ax_compiled = plt.subplots()

    ax_compiled.set_xlabel(r'Rest wavelength [$\mu m$]')
    ax_compiled.set_ylabel('Normalized flux')
    ax_compiled.set_yscale('log')
    ax_compiled.set_ylim(1e-4, 3.)
    ax_compiled.set_xlim(0.5500, 2.3000)

    ax_compiled.axvline(0.8400, color='r', ls='--', zorder=-4)

    ax_compiled.grid(which='both', axis='x', linestyle='-')
    ax_compiled.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_compiled.xaxis.set_minor_locator(MultipleLocator(0.0500))

    for i in range(len(training_flux)):
        flux = training_flux[i]
        flux_err = np.sqrt(training_flux_var[i])

        # Individual plots
        ax.plot(wave, flux, 'k-', zorder=2)
        ax.fill_between(wave, flux - flux_err, flux + flux_err,
                        color='grey', zorder=1)

        ax.set_xlabel('Rest wavelength [A]')
        ax.set_ylabel('Normalized flux')
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 3.)
        ax.set_xlim(5500., 23000.)

        fn = f'{plot_dir}/training_{i + 1}.png'
        fig.savefig(fn, dpi=200)
        ax.clear()

        # Compiled plot
        ax_compiled.plot(wave * 1e-4, flux, '-', lw=1., zorder=2, alpha=0.8,)
        # ax_compiled.fill_between(wave, flux - flux_err, flux + flux_err,
        #                          color='grey', zorder=1, alpha=0.5)

    plt.tight_layout()
    fn = f'{plot_dir}/training_compiled.png'
    fig_compiled.savefig(fn, dpi=200)

    plt.close('all')
