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


fig, ax = plt.subplots()

colors = [choose_color(prob) for prob in probs]
ax.scatter(pc[:, 0], pc[:, 1], s=25., c=colors)

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

for i in range(4):
    col = np.array(color_map[i]) / 255.
    ax.scatter([-1.], [-1.], s=1., c=[col], label=name_map[i])

ax.set_xlim(xlim)
ax.set_ylim(ylim)

plt.tight_layout()

fn = 'branch_pc.pdf'
fig.savefig(fn)
