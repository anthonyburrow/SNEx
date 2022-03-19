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

        self._x_pred = self._get_x_pred(regime, time)
        self._variance = np.zeros_like(self._x_pred)

        # Load model information
        self._model_vectors, self._model_mean, self._model_var = \
            self._load_model(regime, time)

        total_components = len(self._model_vectors)
        if n_components is not None:
            if n_components > len(self._model_vectors):
                msg = (f'n_components must be less than {total_components}.'
                       f' Setting n_components to {total_components}')
                print(msg)
                self.n_components = total_components
            else:
                self.n_components = n_components
        else:
            self.n_components = total_components

        self._model_vectors = self._model_vectors[:self.n_components]

    def fit(self, calc_var=True, *args, **kwargs):
        # Get interpolated flux at PCA wavelengths
        fit_mask = (self.data[0, 0] <= self._x_pred) & \
                   (self._x_pred <= self.data[-1, 0])

        spex = Spextractor(self.data, auto_prune=False, verbose=False)
        interp_flux, interp_var = spex.predict(self._x_pred[fit_mask])
        interp_flux -= self._model_mean[fit_mask]

        # Get eigenvalues (params) for observed region
        fit_vectors = self._model_vectors[:, fit_mask]

        self._params = self._fit_function(interp_flux, fit_vectors)
        if calc_var:
            self._variance = self._calc_variance(interp_flux, interp_var,
                                                 fit_vectors, *args, **kwargs)

    def predict(self, *args, **kwargs):
        y_pred = self._max_flux * (self.function(self._params) + self._model_mean)

        y_err = self._max_flux * np.sqrt(self._model_var + self._variance)

        return y_pred, y_err, self._x_pred

    def function(self, eigenvalues):
        y_pred = self._model_vectors.T @ eigenvalues.T
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

    def _fit_function(self, flux, eigenvectors):
        return eigenvectors @ flux.T

    def _calc_variance(self, flux, var, eigenvectors, var_iter=200,
                       *args, **kwargs):
        print(f'Iterating over {var_iter} PCA fits...')

        # Fit a distribution of spectra sampled from original
        # spectrum uncertainties
        samples = np.random.normal(loc=flux, scale=np.sqrt(var),
                                   size=(var_iter, len(flux)))

        eigenvalues = self._fit_function(eigenvectors, samples)
        predictions = self.function(eigenvalues)

        return predictions.var(axis=1)
