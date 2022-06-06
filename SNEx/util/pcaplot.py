import numpy as np
import matplotlib.pyplot as plt


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
