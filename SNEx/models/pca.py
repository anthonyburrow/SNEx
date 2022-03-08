import numpy as np
import pickle
from spextractor import Spextractor
from os.path import dirname

from ..util.misc import prune_data


available_times = (0, 2, 4, 6)
n_points = 250

nir_model_ranges = {
    0: (5500., 8965.),
    2: (5500., 9075.),
    4: (5500., 9003.),
    6: (5500., 8632.),
}

uv_model_ranges = {
    0: (3687., 5500.),
    2: (3444., 5500.),
    4: (3466., 5500.),
    6: (3732., 5500.),
}


_model_dir = f'{dirname(__file__)}/PCA_models'


class PCA:

    def __init__(self, data, time=None, regime=None, wave_range=None,
                 n_components=None, *args, **kwargs):
        self.data = prune_data(data, wave_range)

        self._time = self._select_time(time) if time is not None else 0
        self._regime = regime if regime is not None else 'nir'

        self._model, self._mean = self._load_model()
        self._n_components = n_components if n_components is not None \
            else self._model.n_components_

        self._max_flux = None
        self._x_pred = None
        self.params = None

    def fit(self, *args, **kwargs):
        if self._regime == 'nir':
            wave_range = nir_model_ranges[self._time]
        elif self._regime == 'uv':
            wave_range == uv_model_ranges[self._time]

        # Get interpolated flux at PCA wavelengths that are in
        # observed spectrum
        self._x_pred = np.linspace(*wave_range, n_points)
        fit_mask = (self.data[0, 0] <= self._x_pred) & \
                   (self._x_pred <= self.data[-1, 0])

        spex = Spextractor(self.data, auto_prune=False, verbose=False)
        self._max_flux = spex.fmax_in
        int_flux, var = spex.predict(self._x_pred[fit_mask])

        # Get eigenvalues (params) for observed region
        fit_vectors = self._model.components_[:self._n_components, fit_mask]

        self.params = fit_vectors @ (int_flux - self._mean[fit_mask])

    def predict(self):
        eigenvectors = self._model.components_[:self._n_components]
        y_pred = (self.params * eigenvectors.T).sum(axis=1)
        return self._x_pred, self._max_flux * (y_pred + self._mean)

    def _select_time(self, time):
        ind = np.abs(np.array(available_times) - time).argmin()
        return available_times[ind]

    def _load_model(self):
        fn = f'{_model_dir}/{self._regime}_{self._time}.pkl'
        with open(fn, 'rb') as file:
            return pickle.load(file)
