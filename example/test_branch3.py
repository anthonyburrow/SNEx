import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from abpy.plot import paper_plot
paper_plot()


fn = './all_params.dat'
data = np.loadtxt(fn)

mask = data[:, 3] == 0.
data = data[~mask]

z = data[:, 0]
sbv = data[:, 1]
time = data[:, 2]
p5 = data[:, 3]
p5_err = data[:, 4]
p6 = data[:, 5]
p6_err = data[:, 6]
vsi = data[:, 7]
vsi_err = data[:, 8]
probs = data[:, 9:13]
pc = data[:, 13:]

# Testing PC reconstruction of 1 SN from each group (the most likely ones)
test_ind = [probs[:, i].argmax() for i in range(4)]
test_data = data[test_ind]
test_pc = test_data[:, 13:]

fn = './eigenvectors.dat'
eigenvectors = np.loadtxt(fn)
fn = './wavelengths.dat'
wavelengths = np.loadtxt(fn)
mean_spec = np.loadtxt('mean.dat')

color_map = {
    # cn, ss, bl, cl
    0: [255, 55, 41],   # red
    1: [102, 245, 59],  # green
    2: [59, 199, 245],  # blue
    3: [202, 59, 245]   # magenta
}

name_map = {
    0: 'CN',
    1: 'SS',
    2: 'BL',
    3: 'CL'
}


def choose_color(prob):
    color = np.array(color_map[prob.argmax()])
    return color / 255.


def make_plt(_ax, pc1, pc2, colors, cmin=None, cmax=None):
    plot_params = {
        's': 10.,
        'c': colors
    }
    if cmin is not None:
        plot_params['cmap'] = 'Spectral'
        plot_params['norm'] = Normalize(vmin=cmin, vmax=cmax)

    scatter = _ax.scatter(pc[:, pc1], pc[:, pc2], **plot_params)
    _ax.tick_params(axis='both', labelsize=6., length=3.)

    plot_params['edgecolor'] = 'k'
    plot_params['linewidths'] = 0.8
    plot_params['c'] = np.array(colors)[test_ind]
    # _ax.scatter(test_pc[:, pc1], test_pc[:, pc2], **plot_params)

    return scatter


def correlation_matrix(n_pcs=4, *args, **kwargs):
    fig, ax = plt.subplots(n_pcs - 1, n_pcs - 1)

    for j in range(0, n_pcs - 1):
        for i in range(0, n_pcs - 1):
            _ax = ax[j, i]
            pc1 = i
            pc2 = j + 1
            if i > j:
                _ax.get_xaxis().set_ticks([])
                _ax.get_yaxis().set_ticks([])
                _ax.spines['top'].set_visible(False)
                _ax.spines['right'].set_visible(False)
                _ax.spines['bottom'].set_visible(False)
                _ax.spines['left'].set_visible(False)
                continue
            scatter = make_plt(_ax, pc1, pc2, *args, **kwargs)

    [ax[-1, i].set_xlabel(f'PC {i + 1}') for i in range(0, n_pcs - 1)]
    [ax[j, 0].set_ylabel(f'PC {j + 2}') for j in range(0, n_pcs - 1)]

    return fig, ax, scatter


# CORRELATION MATRIX - BRANCH GROUPS
colors = [choose_color(prob) for prob in probs]
fig, ax, _ = correlation_matrix(n_pcs=4, colors=colors)

for i in range(4):
    col = np.array(color_map[i]) / 255.
    ax[-2, -1].scatter([-1.], [-1.], s=25., c=[col], label=name_map[i])
ax[-2, -1].set_xlim(1., 2.)
ax[-2, -1].set_ylim(1., 2.)
ax[-2, -1].legend(loc='upper left')

plt.tight_layout()
plt.subplots_adjust(wspace=0.20, hspace=0.2)

fn = 'pcs_branch.pdf'
fig.savefig(fn)
fn = 'pcs_branch.png'
fig.savefig(fn, dpi=200)
plt.close('all')


# PLOT EIGENVECTOR RECONSTRUCTION OF SPECIFIC PCs
fig, ax = plt.subplots()

offset = 0.
which_pc = [1]
for i, _pc in enumerate(test_pc):
    y_pred = (eigenvectors[which_pc].T * _pc[which_pc]).sum(axis=1) + offset
    # y_pred += mean_spec
    col = np.array(color_map[i]) / 255.
    ax.plot(wavelengths, y_pred, color=col, label=name_map[i])
    # offset += 0.3

ax.set_xlabel('Rest wavelength [A]')
ax.set_ylabel('Normalized flux - $\mu$ (Arbitrary units)')

ax.legend()

fn = 'pcs_spectra.pdf'
fig.savefig(fn)
fn = 'pcs_spectra.png'
fig.savefig(fn, dpi=200)
plt.close('all')


# CORRELATION MATRIX - sBV
cmin = sbv.min()
cmax = sbv.max()
fig, ax, scatter = correlation_matrix(n_pcs=4, colors=sbv, cmin=cmin, cmax=cmax)
fig.set_size_inches(7.5, 4.8)

cbar = fig.colorbar(scatter, ax=ax, ticks=[0.5, 0.75, 1.0, 1.25, 1.5])
cbar.ax.tick_params(labelsize=8.)
cbar.set_label(label=r'$\mathrm{s_{BV}}$', size=14.)

plt.tight_layout()
pos0 = cbar.ax.get_position()
cbar.ax.set_position([pos0.x0 + 0.1, pos0.y0, pos0.width, pos0.height])

plt.subplots_adjust(right=0.85)

fn = 'pcs_sbv.pdf'
fig.savefig(fn)
fn = 'pcs_sbv.png'
fig.savefig(fn, dpi=200)
plt.close('all')
