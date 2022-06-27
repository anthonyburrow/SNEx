import numpy as np
import matplotlib.pyplot as plt

from abpy.plot import paper_plot
paper_plot()


fn = './all_params.dat'
data = np.loadtxt(fn)

mask = data[:, 1] == 0.
data = data[~mask]

time = data[:, 0]
p5 = data[:, 1]
p5_err = data[:, 2]
p6 = data[:, 3]
p6_err = data[:, 4]
vsi = data[:, 5]
vsi_err = data[:, 6]
probs = data[:, 7:11]
p_cn = data[:, 7]
p_ss = data[:, 8]
p_bl = data[:, 9]
p_cl = data[:, 10]
pc = data[:, 11:]


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


colors = [choose_color(prob) for prob in probs]


n_pcs = 4
fig, ax = plt.subplots(n_pcs - 1, n_pcs - 1)


def make_plt(_ax, pc1, pc2):
    if pc1 >= pc2:
        _ax.get_xaxis().set_ticks([])
        _ax.get_yaxis().set_ticks([])
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.spines['bottom'].set_visible(False)
        _ax.spines['left'].set_visible(False)
        return

    _ax.scatter(pc[:, pc1], pc[:, pc2], s=10., c=colors)
    _ax.tick_params(axis='both', labelsize=6., length=3.)

    # _ax.set_xlabel(f'PC {pc1 + 1}')
    # _ax.set_ylabel(f'PC {pc2 + 1}')


for j in range(1, n_pcs):
    for i in range(0, n_pcs - 1):
        make_plt(ax[j - 1, i], i, n_pcs - j)

[ax[i - 1, 0].set_ylabel(f'PC {n_pcs - i + 1}') for i in range(1, n_pcs)]
[ax[-1, i].set_xlabel(f'PC {i + 1}') for i in range(0, n_pcs - 1)]

# xlim = _ax.get_xlim()
# ylim = _ax.get_ylim()

for i in range(4):
    col = np.array(color_map[i]) / 255.
    ax[-2, -1].scatter([-1.], [-1.], s=25., c=[col], label=name_map[i])
ax[-2, -1].set_xlim(1., 2.)
ax[-2, -1].set_ylim(1., 2.)
ax[-2, -1].legend(loc='upper left')

plt.tight_layout()
plt.subplots_adjust(wspace=0.20, hspace=0.2)

fn = 'branch_pc.pdf'
fig.savefig(fn)
