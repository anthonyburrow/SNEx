import numpy as np
import pickle
from spextractor import Spextractor
from os.path import dirname

from .extrapolationmodel import ExtrapolationModel


# Model information
_available_times = (0, 2, 4, 6)
_n_points = 250

_nir_model_ranges = {
    0: (5500., 8965.),
    2: (5500., 9075.),
    4: (5500., 9003.),
    6: (5500., 8632.),
}

_uv_model_ranges = {
    0: (3687., 5500.),
    2: (3444., 5500.),
    4: (3466., 5500.),
    6: (3732., 5500.),
}


_model_dir = f'{dirname(__file__)}/PCA_models'


class PCA(ExtrapolationModel):

    def __init__(self, regime, time=None, n_components=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        time = self._select_model_time(time)

        self._model, self._model_mean = self._load_model(regime, time)
        self.n_components = self._model.n_components_ if n_components is None \
            else n_components

        self._x_pred = self._get_x_pred(regime, time)

    def fit(self, *args, **kwargs):
        # Get interpolated flux at PCA wavelengths
        fit_mask = (self.data[0, 0] <= self._x_pred) & \
                   (self._x_pred <= self.data[-1, 0])

        spex = Spextractor(self.data, auto_prune=False, verbose=False)
        interp_flux, var = spex.predict(self._x_pred[fit_mask])

        # Get eigenvalues (params) for observed region
        fit_vectors = self._model.components_[:self.n_components, fit_mask]

        self._params = fit_vectors @ (interp_flux - self._model_mean[fit_mask])

    def predict(self, *args, **kwargs):
        return self._max_flux * self.function(self._params), self._x_pred

    def function(self, eigenvalues):
        eigenvectors = self._model.components_[:self.n_components]
        y_pred = (eigenvalues * eigenvectors.T).sum(axis=1)
        y_pred += self._model_mean
        return y_pred

    def __str__(self):
        return ''

    def _load_model(self, regime, time):
        fn = f'{_model_dir}/{regime}_{time}.pkl'
        with open(fn, 'rb') as file:
            return pickle.load(file)

    def _get_x_pred(self, regime, time):
        if regime == 'nir':
            wave_range = _nir_model_ranges[time]
        elif regime == 'uv':
            wave_range == _uv_model_ranges[time]

        return np.linspace(*wave_range, _n_points)

    def _select_model_time(self, time):
        if time is None:
            return 0

        ind = np.abs(np.array(_available_times) - time).argmin()
        return _available_times[ind]
