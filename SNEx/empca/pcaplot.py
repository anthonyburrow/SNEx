import numpy as np
import matplotlib.pyplot as plt


def plot_info(pcamodel):
    plot_eigenvectors(pcamodel)
    plot_explained_var(pcamodel)
    plot_training(pcamodel)
    plt.close('all')


def plot_eigenvectors(pcamodel):
    fig, ax = plt.subplots()

    wave = pcamodel.wave
    eig = pcamodel.eigenvectors

    n_comp = 3
    for i in range(n_comp):
        ax.plot(wave, eig[i], '-', label=f'PC{i + 1}', zorder=n_comp - i)

    ax.set_xlabel('Rest wavelength (A)')
    ax.set_ylabel(r'$F - F_\mu$')

    ax.axhline(0., color='k', ls='--', zorder=-4)

    ax.legend()

    plt.tight_layout()

    plot_dir = './pca_info'
    fn = f'{plot_dir}/eigenvectors.pdf'
    fig.savefig(fn)
    fn = f'{plot_dir}/eigenvectors.png'
    fig.savefig(fn, dpi=200)


def plot_explained_var(pcamodel):
    fig, ax = plt.subplots()

    pc_ids = np.arange(1, pcamodel.n_components + 1)
    ax.plot(pc_ids, pcamodel.explained_var, 'ko')

    ax.set_yscale('log')
    ax.set_ylim(top=1.)

    ax.set_xlabel('PC')
    ax.set_ylabel('fractional explained variance')

    plt.tight_layout()

    plot_dir = './pca_info'
    fn = f'{plot_dir}/explained_variance.pdf'
    fig.savefig(fn)
    fn = f'{plot_dir}/explained_variance.png'
    fig.savefig(fn, dpi=200)


def plot_training(pcamodel):
    plot_dir = './training_spec'

    wave = pcamodel.wave
    training_flux = pcamodel.training_flux
    training_flux_var = pcamodel.flux_var_train
    training_times = pcamodel.training_times

    fig, ax = plt.subplots()
    fig_compiled, ax_compiled = plt.subplots()

    ax_compiled.set_xlabel('Rest wavelength [A]')
    ax_compiled.set_ylabel('Normalized flux')
    ax_compiled.set_yscale('log')
    ax_compiled.set_ylim(3e-2, 6.)
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
        ax.set_ylim(3e-2, 6.)
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
