import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from ..util.plot_setup import paper_plot
paper_plot()


def plot_info(pcamodel):
    plot_eigenvectors(pcamodel)
    plot_explained_var(pcamodel)
    plot_explained_var_cumsum(pcamodel)
    plot_training(pcamodel)
    plt.close('all')


def plot_eigenvectors(pcamodel):
    n_comp = 8
    fig, ax = plt.subplots(n_comp, 1, sharey=True, sharex=True, figsize=(6, 8))

    wave = pcamodel.wave * 1e-4
    eig = pcamodel.eigenvectors

    for i, _ax in enumerate(ax):
        _ax.plot(wave, eig[i], '-', label=f'PC{i + 1}', c='tab:blue')
        _ax.axhline(0., color='k', ls='--', zorder=-4)
        _ax.text(0.9, 0.7, f'PC{i + 1}', transform=_ax.transAxes, fontsize=12)

    ax[-1].set_xlabel(r'Rest wavelength ($\mu m$)')
    # [_ax.set_ylabel(r'$(F - \mu_F) / \sigma_F$', fontsize=10) for _ax in ax]
    plt.ylabel(r'$(F - \mu_F) / \sigma_F$', fontsize=10)

    ax[0].xaxis.set_minor_locator(MultipleLocator(0.05))
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.01))
    ax[0].yaxis.set_major_locator(MultipleLocator(0.04))

    [_ax.tick_params(axis='both', which='major', labelsize=10) for _ax in ax]

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
    ax.plot(pc_ids, pcamodel.explained_var.cumsum(), 'ko', ms=6.)

    ax.axhline(0.95, c='r', ls='--')

    ax.set_ylim(0., 1.)

    ax.set_xlabel('PC')
    ax.set_ylabel('Fractional explained variance')

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
    training_times = pcamodel.training_times

    training_flux, training_flux_var = pcamodel.descale(training_flux,
                                                        training_flux_var)

    fig, ax = plt.subplots()
    fig_compiled, ax_compiled = plt.subplots()

    ax_compiled.set_xlabel('Rest wavelength [A]')
    ax_compiled.set_ylabel('Normalized flux')
    ax_compiled.set_yscale('log')
    # ax_compiled.set_ylim(3e-2, 6.)
    ax_compiled.set_xlim(5500., 11000.)

    ax_compiled.axvline(8400., color='k', ls='--', zorder=-4)

    which_time = 0
    times = training_times[:, which_time]
    times = np.abs(training_times[:, 0] - training_times[:, 1])
    min_time = times.min()
    max_time = times.max()
    times_norm = (times - min_time) / (max_time - min_time)
    colors = plt.cm.Spectral(times_norm)
    sm = plt.cm.ScalarMappable(cmap='Spectral',
                               norm=plt.Normalize(vmin=min_time, vmax=max_time))
    cbar = fig_compiled.colorbar(sm)
    cbar.set_label(label=r'time past $B_{max}$', size=14.)
    # cbar.set_label(label=r'$|t_{CSP} - t_{FIRE}|$', size=14.)

    for i in range(len(training_flux)):
        flux = training_flux[i]
        flux_err = np.sqrt(training_flux_var[i])

        # Individual plots
        ax.plot(wave, flux, 'k-', zorder=2)
        ax.fill_between(wave, flux - flux_err, flux + flux_err, color='grey',
                        zorder=1)

        ax.set_xlabel('Rest wavelength [A]')
        ax.set_ylabel('Normalized flux')
        ax.set_yscale('log')
        # ax.set_ylim(3e-2, 6.)
        ax.set_xlim(5500., 11000.)

        fn = f'{plot_dir}/training_{i}.png'
        fig.savefig(fn, dpi=200)
        ax.clear()

        # Compiled plot
        ax_compiled.plot(wave, flux, '-', c=colors[i], lw=1., zorder=2,
                         alpha=0.8,)
        # ax_compiled.fill_between(wave, flux - flux_err, flux + flux_err,
        #                          color='grey', zorder=1, alpha=0.5)

    plt.tight_layout()
    fn = f'{plot_dir}/training_compiled.png'
    fig_compiled.savefig(fn, dpi=200)

    plt.close('all')
