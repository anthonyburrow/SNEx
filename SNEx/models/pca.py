import numpy as np
from spextractor import Spextractor

from .extrapolationmodel import ExtrapolationModel
from ..PCA.load import load_model
from ..PCA.pcaplot import plot_info
from ..util.misc import setup_clean_dir


class PCA(ExtrapolationModel):

    def __init__(self, n_components=None, plot_pca=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self._prune_fit_features(*args, **kwargs)

        # Load model information
        self._model = load_model(*args, **kwargs)
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

        self._save_info(plot=plot_pca)

    def fit(self, calc_var=True, *args, **kwargs):
        # Interpolate data to get on the same waves as eigenvectors
        fit_mask = (self.data[0, 0] <= self._model.wave) & \
                   (self._model.wave <= self.data[-1, 0])

        spex = Spextractor(self.data, auto_prune=False, verbose=False)
        interp_flux, interp_var = spex.predict(self._model.wave[fit_mask])
        interp_flux *= spex.fmax_out
        interp_var *= spex.fmax_out**2

        # Normalize interpolation the same way
        # norm = get_normalization(interp_flux, *args, **kwargs)
        # interp_flux /= norm
        # interp_var /= norm**2

        # Scale (subtract mean) from interpolation
        interp_flux, interp_var = self._model.scale(interp_flux, interp_var,
                                                    fit_mask)

        # Get eigenvalues (params) for observed region
        fit_vectors = self._model.eigenvectors[:self.n_components, fit_mask]

        self._params = self._fit_function(interp_flux, fit_vectors)

        if not calc_var:
            return

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

        y_pred *= self._norm
        y_err = self._norm * np.sqrt(y_var_pred)

        return y_pred, y_err, self._model.wave

    def function(self, eigenvalues):
        y_pred = (eigenvalues @ self._model.eigenvectors[:self.n_components]).T
        return y_pred

    def __str__(self):
        return ''

    # def _prune_fit_features(self, fit_features=None, *args, **kwargs):
    #     if fit_features is None:
    #         return
    #     if not fit_features:
    #         return

    #     wave = self.data[:, 0]
    #     mask = np.full(len(self.data), False)
    #     for feature in fit_features:
    #         wave_range = feature_ranges[feature]
    #         sub_mask = \
    #             (wave_range[0] <= wave) & (wave <= wave_range[1])
    #         mask += sub_mask

    #     self.data = self.data[mask]
    #     self._norm = get_normalization(self.data[:, 1], *args, **kwargs)

    def _fit_function(self, flux, eigenvectors):
        return eigenvectors @ flux.T

    def _calc_variance(self, flux, var, eigenvectors, var_iter=500,
                       *args, **kwargs):
        msg = f'Iterating over {var_iter} PCA fits to estimate uncertainty...'
        print(msg)

        # Fit a distribution of spectra sampled from original
        # spectrum uncertainties
        samples = np.random.normal(loc=flux, scale=np.sqrt(var),
                                   size=(var_iter, len(flux)))

        eigenvalues = self._fit_function(eigenvectors, samples)
        predictions = self.function(eigenvalues)

        return predictions.var(axis=1)

    def _save_info(self, plot):
        info_dir = './pca_info'
        setup_clean_dir(info_dir)

        fn = f'{info_dir}/eigenvectors.dat'
        np.savetxt(fn, self._model.eigenvectors[:self.n_components])
        fn = f'{info_dir}/wavelengths.dat'
        np.savetxt(fn, self._model.wave)
        fn = f'{info_dir}/mean.dat'
        np.savetxt(fn, self._model.mean)
        fn = f'{info_dir}/std.dat'
        np.savetxt(fn, self._model.std)
        fn = f'{info_dir}/flux_train.dat'
        np.savetxt(fn, self._model.flux_train)
        fn = f'{info_dir}/flux_var_train.dat'
        np.savetxt(fn, self._model.flux_var_train)

        if plot:
            plot_dir = './training_spec'
            setup_clean_dir(plot_dir)
            plot_info(self._model)
