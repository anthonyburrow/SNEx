import wave
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from spextractor import Spextractor
from SNIaDCA import GMM
from spextractor.physics.feature import pEW


fn = './spec_list'
dt = [('name', 'U16'), ('time', np.float64)]
spec_list = np.loadtxt(fn, dtype=dt)


try:
    _snexgen_dir = f'{Path.home()}/dev/SNEx_gen'
    if not os.path.isdir(_snexgen_dir):
        raise FileNotFoundError
except FileNotFoundError:
    _snexgen_dir = f'C:/dev/SNEx_gen'

_spex_interp_dir = f'{_snexgen_dir}/model_scripts/time_interp/spex/csp'


siII_6355 = 'Si II 6150A'
siII_5972 = 'Si II 5800A'

total_wave_range = (3500., 10000.)
total_n_points = 2000
total_wave = np.linspace(*total_wave_range, total_n_points)

wave_range = (4000., 8000.)
wave_mask = (wave_range[0] < total_wave) & (total_wave < wave_range[1])

total_wave = total_wave[wave_mask]
data = np.zeros((len(total_wave), 3))
data[:, 0] = total_wave

pEW_5972 = []
pEW_6355 = []
for sn in spec_list:
    name = sn['name']
    time = sn['time']
    sn_fn = f'{_spex_interp_dir}/{name}/model_{time:.3f}.dat'

    data[:, 1:3] = np.loadtxt(sn_fn)[wave_mask]

    spex = Spextractor(data, verbose=False)
    spex.create_model(downsampling=3., sigma_outliers=-1.)
    spex.process(features=(siII_5972, siII_6355))

    pEW_5972.append((spex.pew[siII_5972], spex.pew_err[siII_5972]))
    pEW_6355.append((spex.pew[siII_6355], spex.pew_err[siII_6355]))

pEW_5972 = np.array(pEW_5972)
pEW_6355 = np.array(pEW_6355)

nan_mask = ~(np.isnan(pEW_5972).any(axis=1) + np.isnan(pEW_6355).any(axis=1))
pEW_5972 = pEW_5972[nan_mask]
pEW_6355 = pEW_6355[nan_mask]
spec_list = spec_list[nan_mask]

model = GMM(pew_5972=pEW_5972[:, 0], pew_6355=pEW_6355[:, 0])
probs = model.predict()

for i in range(len(spec_list)):
    group_name, group_prob = model.get_group_name(probs[i])
    print(f'{spec_list["name"][i]} ({spec_list["time"][i]}) : '
          f'{group_name} ({group_prob * 100.:.2f}%)')


# SN2014D (3.875) : Core-normal (93.91%)
# SN2012ij (-0.178) : Cool (97.40%)
# ASAS15bm (1.171) : Shallow-silicon (97.74%)
