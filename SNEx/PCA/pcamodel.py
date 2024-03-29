import numpy as np

from .empca import empca
from ..util.misc import between_mask


_telluric_regions = ((12963., 14419.), (17421., 19322.))
_empca_weight_method = 'normalized_variance'


class PCAModel:

    def __init__(self, wave, flux_train, flux_var_train=None,
                 n_components=None, *args, **kwargs):
        self.wave = wave
        self.flux_train = flux_train
        self.flux_var_train = flux_var_train

        self.n_components = n_components
        self.n_points = len(wave)

        self.mean = self.flux_train.mean(axis=0)
        self.std = self.flux_train.std(axis=0)
        self.flux_train, self.flux_var_train = self.scale(self.flux_train,
                                                          self.flux_var_train)

        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_var = None

    def descale(self, flux, flux_var, mask=None):
        mean = self.mean if mask is None else self.mean[mask]
        std = self.std if mask is None else self.std[mask]

        new_flux = (flux * std) + mean
        new_flux_var = flux_var * std**2

        return new_flux, new_flux_var

    def scale(self, flux, flux_var, mask=None):
        mean = self.mean if mask is None else self.mean[mask]
        std = self.std if mask is None else self.std[mask]

        new_flux = (flux - mean) / std
        new_flux_var = flux_var / std**2

        return new_flux, new_flux_var

    def calc_eig(self):
        weighted = not np.all(self.flux_var_train == 0.) or \
            self.flux_var_train is None

        weight_method = _empca_weight_method
        if not weighted:
            weight_method = 'uniform'
            print('No uncertainty detected; using equal PCA weights.')

        weights = self._calc_empca_weights(weight_method=weight_method)
        pca = empca(self.flux_train, weights=weights, niter=25,
                    nvec=self.n_components, silent=True, seed=2)
        self.eigenvalues = pca.coeff
        self.eigenvectors = pca.eigvec
        self.explained_var = [pca.R2vec(i) for i in range(self.n_components)]
        self.explained_var = np.array(self.explained_var)

    def calc_var(self, fit_mask, n_components):
        # Get eigenvalues inside fitting region
        fit_flux = self.flux_train[:, fit_mask]
        fit_eigenvectors = self.eigenvectors[:n_components, fit_mask]

        fit_eigenvalues = fit_eigenvectors @ fit_flux.T

        # Predict everywhere, then see how this varies from training data
        y_pred = (self.eigenvectors[:n_components].T @ fit_eigenvalues).T
        residual = self.flux_train - y_pred
        model_var = residual.var(axis=0)

        return model_var

    def _calc_empca_weights(self, weight_method=None):
        if weight_method is None or weight_method == 'uniform':
            return np.ones(self.flux_var_train.shape)
        elif weight_method == 'telluric':
            weights = np.ones(self.flux_var_train.shape)
            for wave_range in _telluric_regions:
                mask = between_mask(self.wave, wave_range)
                min_var = self.flux_var_train[:, mask].min(axis=0)
                max_var = self.flux_var_train[:, mask].max(axis=0)
                norm_var = (self.flux_var_train[:, mask] - min_var) / (max_var - min_var)
                weights[:, mask] = 1. - norm_var
            return weights
        elif weight_method == 'inverse_variance':
            return 1. / self.flux_var_train
        elif weight_method == 'normalized_variance':
            min_var = self.flux_var_train.min(axis=0)
            max_var = self.flux_var_train.max(axis=0)
            # norm_var = (self.flux_var_train - min_var) / (max_var - min_var)
            norm_var = self.flux_var_train / max_var

            scale_range = 0.0, 0.75
            norm_var *= (scale_range[1] - scale_range[0]) + scale_range[0]

            return 1. - norm_var
        elif weight_method == 'normalized_variance_2':
            min_var = self.flux_var_train.min(axis=0)
            return min_var / self.flux_var_train
