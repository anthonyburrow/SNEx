import matplotlib.pyplot as plt


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
    # fig, ax = plt.subplots()


    # fn = './explained_variance.pdf'
    # fig.savefig(fn)
    pass