import numpy as np
from sklearn.decomposition import PCA
import empca

from .misc import get_normalization


class PCAModel:

    def __init__(self, wave, flux_train, flux_var_train=None,
                 n_components=None, *args, **kwargs):
        self.wave = wave
        self.flux_train = flux_train
        self.flux_var_train = flux_var_train

        self.n_components = n_components
        self.n_points = len(wave)

        self.mean = self.flux_train.mean(axis=0)
        self.flux_train, self.flux_var_train = self.scale(self.flux_train,
                                                          self.flux_var_train)

        np.savetxt('mean.dat', self.mean)

        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_var = None

    def descale(self, flux, flux_var, mask=None):
        mean = self.mean if mask is None else self.mean[mask]

        new_flux = flux + mean
        new_flux_var = flux_var

        return new_flux, new_flux_var

    def scale(self, flux, flux_var, mask=None):
        mean = self.mean if mask is None else self.mean[mask]

        new_flux = flux - mean
        new_flux_var = flux_var

        return new_flux, new_flux_var

    def calc_eig(self):
        weighted = not np.all(self.flux_var_train == 0.) or \
            self.flux_var_train is None

        weighted = False
        if weighted:
            print('Uncertainty detected; generating EMPCA model')
            weights = 1. / self.flux_var_train
            pca = empca.empca(self.flux_train, weights=weights, niter=25,
                              nvec=self.n_components, silent=True)
            self.eigenvalues = pca.coeff
            self.eigenvectors = pca.eigvec
            self.explained_var = [pca.R2vec(i) for i in range(self.n_components)]
            self.explained_var = np.array(self.explained_var)
        else:
            print('No uncertainty detected; generating PCA model')
            pca = PCA(n_components=self.n_components)
            self.eigenvalues = pca.fit_transform(self.flux_train)
            self.eigenvectors = pca.components_
            self.explained_var = pca.explained_variance_ratio_

    def calc_var(self, fit_mask, n_components):
        # Get eigenvalues inside fitting region
        fit_flux = self.flux_train[:, fit_mask]
        fit_eigenvectors = self.eigenvectors[:n_components, fit_mask]

        fit_eigenvalues = fit_eigenvectors @ fit_flux.T

        # Predict everywhere, then see how this varies from training data
        y_pred = (self.eigenvectors[:n_components].T @ fit_eigenvalues).T
        model_var = (self.flux_train - y_pred).var(axis=0)

        return model_var
