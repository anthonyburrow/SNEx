import numpy as np
import pickle
import os


_model_save_dir = f'{os.path.dirname(__file__)}/PCA_models'


def get_time(fn):
    time = os.path.splitext(fn)[0]
    time = time.split('_')[-1]
    time = float(time)
    return time


def load_model(time, *args, **kwargs):
    models = os.listdir(_model_save_dir)
    times = np.array([get_time(fn) for fn in models])

    ind = np.abs(times - time).argmin()
    model_time = times[ind]
    model_fn = models[ind]
    model_fn = f'{_model_save_dir}/{model_fn}'

    print(f'Loading [Day {model_time:.1f}] PCA model')

    return pickle.load(open(model_fn, 'rb'))
