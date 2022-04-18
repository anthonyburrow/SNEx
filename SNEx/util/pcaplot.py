import matplotlib.pyplot as plt


def plot_eigenvectors(pcamodel):
    fig, ax = plt.subplots()

    wave = pcamodel.wave
    eig = pcamodel.eigenvectors

    for i in range(4):
        ax.plot(wave, eig[i], '-', label=f'PC{i + 1}')

    fn = './eigenvectors.pdf'
    fig.savefig(fn)


def plot_explained_var(pcamodel):
    # fig, ax = plt.subplots()


    # fn = './explained_variance.pdf'
    # fig.savefig(fn)
    pass