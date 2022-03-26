import numpy as np
import pickle
from spextractor import Spextractor
from os.path import dirname

from .extrapolationmodel import ExtrapolationModel


# Model information
_available_times = (0, 2, 4, 6)

_model_dir = f'{dirname(__file__)}/PCA_models'


class PCA(ExtrapolationModel):

    def __init__(self, regime, time=None, n_components=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        time = self._select_model_time(time)

        # Load model information
        self._model = self._load_model(regime, time)
        self._variance = np.zeros(self._model.n_points)

        total_components = self._model.n_components
        if n_components is not None:
            if n_components > total_components:
                msg = (f'n_components cannot be greater than {total_components}.'
                       f' Setting n_components to {total_components}')
                print(msg)
                self.n_components = total_components
            else:
                self.n_components = n_components
        else:
            self.n_components = total_components

        self._model_vectors = self._model.eigenvectors[:self.n_components]

    def fit(self, calc_var=True, *args, **kwargs):
        # Get interpolated flux at PCA wavelengths
        fit_mask = (self.data[0, 0] <= self._model.wave) & \
                   (self._model.wave <= self.data[-1, 0])

        spex = Spextractor(self.data, auto_prune=False, verbose=False)
        interp_flux, interp_var = spex.predict(self._model.wave[fit_mask])
        interp_flux, interp_var = self._model.scale(interp_flux, interp_var,
                                                    fit_mask)

        # Get eigenvalues (params) for observed region
        fit_vectors = self._model_vectors[:, fit_mask]

        self._params = self._fit_function(interp_flux, fit_vectors)
        if calc_var:
            # Variance from model's ability to predict training data outside
            # the fit mask with n_components eigenvectors
            model_var = self._model.calc_var(fit_mask, self.n_components)

            # Variance due to uncertainty in data causing potential differences
            # in fitting parameters (eigenvalues)
            data_var = self._calc_variance(interp_flux, interp_var,
                                           fit_vectors, *args, **kwargs)

            self._variance = model_var + data_var

    def predict(self, *args, **kwargs):
        y_pred, y_var_pred = \
            self._model.descale(self.function(self._params), self._variance)
        y_pred *= self._max_flux
        y_err = self._max_flux * np.sqrt(y_var_pred)

        # y_pred = self._max_flux * (self.function(self._params) + self._model_mean)
        # y_err = self._max_flux * np.sqrt(self._model_var + self._variance)

        return y_pred, y_err, self._model.wave

    def function(self, eigenvalues):
        y_pred = self._model_vectors.T @ eigenvalues.T
        return y_pred

    def __str__(self):
        return ''

    def _load_model(self, regime, time):
        fn = f'{_model_dir}/{regime}_{time}.pkl'
        with open(fn, 'rb') as file:
            return pickle.load(file)

    def _select_model_time(self, time):
        if time is None:
            return 0

        ind = np.abs(np.array(_available_times) - time).argmin()
        return _available_times[ind]

    def _fit_function(self, flux, eigenvectors):
        return eigenvectors @ flux.T

    def _calc_variance(self, flux, var, eigenvectors, var_iter=1000,
                       *args, **kwargs):
        print(f'Iterating over {var_iter} PCA fits...')

        # Fit a distribution of spectra sampled from original
        # spectrum uncertainties
        samples = np.random.normal(loc=flux, scale=np.sqrt(var),
                                   size=(var_iter, len(flux)))

        eigenvalues = self._fit_function(eigenvectors, samples)
        predictions = self.function(eigenvalues)

        return predictions.var(axis=1)
