import numpy as np
import matplotlib.pyplot as plt

from .misc import setup_clean_dir


def plot_info(pcamodel):
    plot_eigenvectors(pcamodel)
    plot_explained_var(pcamodel)
    plt.close('all')


def plot_eigenvectors(pcamodel):
    fig, ax = plt.subplots()

    wave = pcamodel.wave
    eig = pcamodel.eigenvectors

    n_comp = 4
    for i in range(n_comp):
        ax.plot(wave, eig[i], '-', label=f'PC{i + 1}', zorder=n_comp - i)

    ax.set_xlabel('Rest wavelength (A)')
    ax.set_ylabel(r'$F - F_\mu$')

    ax.legend()

    plt.tight_layout()
    fn = './eigenvectors.pdf'
    fig.savefig(fn)


def plot_explained_var(pcamodel):
    fig, ax = plt.subplots()

    pc_ids = np.arange(1, pcamodel.n_components + 1)
    ax.plot(pc_ids, pcamodel.explained_var, 'ko')

    ax.set_yscale('log')
    ax.set_ylim(top=1.)

    ax.set_xlabel('PC')
    ax.set_ylabel('fractional explained variance')

    plt.tight_layout()
    fn = './explained_variance.pdf'
    fig.savefig(fn)


def plot_training(wave, training_flux, training_flux_var):
    plot_dir = './training_spec'
    setup_clean_dir(plot_dir)

    fig, ax = plt.subplots()

    for i in range(len(training_flux)):
        flux = training_flux[i]
        flux_err = np.sqrt(training_flux_var[i])

        ax.plot(wave, flux, 'k-', zorder=2)
        ax.fill_between(wave, flux - flux_err, flux + flux_err, color='grey',
                        zorder=1)

        fn = f'{plot_dir}/training_{i}.png'
        fig.savefig(fn)
        ax.clear()
    
    plt.close('all')
