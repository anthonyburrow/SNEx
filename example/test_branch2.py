from wave import Wave_read
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from spextractor import Spextractor
from SNIaDCA import GMM


try:
    _snexgen_dir = f'{Path.home()}/dev/SNEx_gen'
    if not os.path.isdir(_snexgen_dir):
        raise FileNotFoundError
except FileNotFoundError:
    _snexgen_dir = f'C:/dev/SNEx_gen'

_spex_interp_dir = f'{_snexgen_dir}/model_scripts/time_interp/spex/csp'

total_wave_range = (3500., 10000.)
total_n_points = 2000
total_wave = np.linspace(*total_wave_range, total_n_points)

wave_range = (5000., 7000.)
pc_wave_range = (5700., 6400.)

area_mask = (wave_range[0] < total_wave) & (total_wave < wave_range[1])
total_wave = total_wave[area_mask]
si_mask = (pc_wave_range[0] < total_wave) & (total_wave < pc_wave_range[1])

data = np.zeros((len(total_wave), 3))
data[:, 0] = total_wave

siII_6355 = 'Si II 6150A'
siII_5972 = 'Si II 5800A'


fn = './wavelengths.dat'
model_wave = np.loadtxt(fn)
model_mask = (pc_wave_range[0] < model_wave) & (model_wave < pc_wave_range[1])
model_wave = model_wave[model_mask]

fn = './eigenvectors.dat'
eigenvectors = np.loadtxt(fn)
eigenvectors = eigenvectors[:, model_mask]

time_window = 5.

def get_params(name, time):
    fn = f'{_spex_interp_dir}/{name}/model_{time}.dat'
    data[:, 1:3] = np.loadtxt(fn)[area_mask]

    # Get Si II pEWs and velocity
    spex = Spextractor(data)
    spex.create_model(sigma_outliers=-1)
    spex.process(features=(siII_5972, siII_6355))

    # params = [str(name), float(time)]
    params = [float(time)]

    params.append(spex.pew[siII_5972])
    params.append(spex.pew_err[siII_5972])
    params.append(spex.pew[siII_6355])
    params.append(spex.pew_err[siII_6355])
    params.append(spex.vel[siII_6355])
    params.append(spex.vel_err[siII_6355])

    # Get GMM probabilities
    model = GMM(pew_5972=np.array([spex.pew[siII_5972]]),
                pew_6355=np.array([spex.pew[siII_6355]]))
    probs = model.predict()[0]
    for prob in probs:
        params.append(prob)

    # Get eigenvalues
    eigenvalues = eigenvectors @ data[:, 1][si_mask]
    for eig in eigenvalues:
        params.append(eig)

    print(params)
    print(len(params))
    return params


def file_to_time(fn):
    return float(fn[6:-4])


all_params = []
test = 0
for sn in os.listdir(_spex_interp_dir):
    spec_files = os.listdir(f'{_spex_interp_dir}/{sn}')

    spec_times = [file_to_time(fn) for fn in spec_files]
    spec_times = np.array(spec_times)
    if not np.any(abs(spec_times) < time_window):
        continue

    closest_ind = np.abs(spec_times).argmin()
    time = spec_times[closest_ind]

    # name, time, p5, p5e, p6, p6e, vsi, vsie, p(cn), p(ss), p(bl), p(cl), PCs
    all_params.append(get_params(sn, time))

    test += 1
    if test > 1:
        break

fn = './all_params.dat'
dt = []
for i in range(len(all_params[0])):
    dt.append('%.8f')
head = 'name, time, p5, p5e, p6, p6e, vsi, vsie, p(cn), p(ss), p(bl), p(cl), PCs'
np.savetxt(fn, all_params, fmt=dt, header=head)
