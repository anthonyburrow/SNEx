"""
Weighted Principal Component Analysis using Expectation Maximization

This is a slightly optimized version of EMPCA originally from Stephen Bailey,
Spring 2012. Optimizations come mostly from vectorizing the eigenvector
update function.

Source: https://github.com/sbailey/empca
"""
import numpy as np
from scipy.signal import savgol_filter
import sys

from .util import norm, random_orthonormal


class Model:
    """
    A wrapper class for storing data, eigenvectors, and coefficients.

    Returned by empca() function. Useful member variables:
      Inputs:
        - eigvec [nvec, nvar]
        - data   [nobs, nvar]
        - weights[nobs, nvar]

      Calculated from those inputs:
        - coeff  [nobs, nvec] - coeffs to reconstruct data using eigvec
        - model  [nobs, nvar] - reconstruction of data using eigvec,coeff

    Not yet implemented: eigenvalues, mean subtraction/bookkeeping
    """

    def __init__(self, data, weights, nvec, *args, **kwargs):
        """
        Create a Model object with eigenvectors, data, and weights.

        Dimensions:
          - eigvec [nvec, nvar]  = [k, j]
          - data   [nobs, nvar]  = [i, j]
          - weights[nobs, nvar]  = [i, j]
          - coeff  [nobs, nvec]  = [i, k]
        """
        self.nobs, self.nvar = data.shape
        self.nvec = nvec

        self.eigvec = random_orthonormal(self.nvec, self.nvar, *args, **kwargs)

        self.data = data
        self.weights = weights

        self.coeff = np.zeros((self.nobs, self.nvec))
        self.model = None

        # Calculate degrees of freedom
        mask = self.weights > 0.
        self.dof = self.data[mask].size - self.eigvec.size - self.nvec * self.nobs

        # Cache variance of unmasked data
        self._unmasked = mask
        self._unmasked_data_var = np.var(self.data[mask])

        self.solve_coeffs()

    def solve_coeffs(self):
        """
        Solve for c[i,k] such that data[i] ~= Sum_k: c[i,k] eigvec[k]
        """
        self.model = None

        weighted_mask = np.all(self.weights.T[1:] == self.weights.T[0], axis=0)

        # Projections for rows with equal weights
        # coeff = (eigvec @ data.T).T = data @ eigvec.T
        self.coeff[weighted_mask] = self.data[weighted_mask] @ self.eigvec.T

        # Projections for rows without equal weights
        for i in range(self.nobs):
            if weighted_mask[i]:
                continue
            self.coeff[i] = self._solve_coeffs_weighted(self.eigvec,
                                                        self.data[i],
                                                        self.weights[i])

    def solve_eigenvectors(self, smooth=None, *args, **kwargs):
        """
        Solve for eigvec[k,j] such that data[i] = Sum_k: coeff[i,k] eigvec[k]
        """
        self.model = None
        data = self.data.copy()

        for k in range(self.nvec):
            c = self.coeff[:, k]
            cw = self.weights.T * c
            self.eigvec[k] = (cw.T * data).sum(axis=0) / (cw * c).sum(axis=1)

            if smooth is not None and smooth > 0.:
                self.eigvec[k] = \
                    savgol_filter(self.eigvec[k], window_length=smooth,
                                  polyorder=3)

            data -= np.outer(self.coeff[:, k], self.eigvec[k])

        # Renormalize and re-orthogonalize via Gram-Schmidt
        self.eigvec[0] /= norm(self.eigvec[0])
        for k in range(1, self.nvec):
            for kx in range(0, k):
                c = self.eigvec[k] @ self.eigvec[kx]
                self.eigvec[k] -= c * self.eigvec[kx]
            self.eigvec[k] /= norm(self.eigvec[k])

    def model(self):
        if self.model is None:
            self.model = np.zeros(self.data.shape)
            self.update_model()

        return self.model

    def update_model(self):
        """
        Uses eigenvectors and coefficients to model data
        """
        self.model = (self.coeff @ self.eigvec).T

    def chi2(self):
        """
        Returns sum( (model-data)^2 / weights )
        """
        delta = (self.model - self.data) * np.sqrt(self.weights)
        return np.sum(delta**2)

    def rchi2(self):
        """
        Returns reduced chi2 = chi2/dof
        """
        return self.chi2() / self.dof

    def _model_vec(self, i):
        """Return the model using just eigvec i"""
        return np.outer(self.coeff[:, i], self.eigvec[i])

    def R2vec(self, i):
        """
        Return fraction of data variance which is explained by vector i.

        Notes:
          - Does *not* correct for degrees of freedom.
          - Not robust to data outliers.
        """

        d = self._model_vec(i) - self.data
        return 1.0 - d[self._unmasked].var() / self._unmasked_data_var

    def R2(self, nvec=None):
        """
        Return fraction of data variance which is explained by the first
        nvec vectors.  Default is R2 for all vectors.

        Notes:
          - Does *not* correct for degrees of freedom.
          - Not robust to data outliers.
        """
        if nvec is None:
            mx = self.model
        else:
            mx = np.zeros(self.data.shape)
            for i in range(nvec):
                mx += self._model_vec(i)

        d = mx - self.data

        # Only consider R2 for unmasked data
        return 1. - d[self._unmasked].var() / self._unmasked_data_var

    def _solve_coeffs_weighted(self, eigvec, data, weights):
        """
        Solve eigvec @ x = data with weights; return x = coeffs

        eigvec : 2D array
        data : 1D array length nvar
        weights : 1D array length nvar
        """
        data = eigvec @ (weights * data)
        eigvec = eigvec @ (eigvec * weights).T

        coeff, _1, _2, _3 = np.linalg.lstsq(eigvec, data, rcond=None)

        return coeff


def empca(data, weights=None, niter=25, nvec=5, silent=False, *args, **kwargs):
    """
    Iteratively solve data[i] = Sum_j: c[i,j] p[j] using weights

    Input:
      - data[nobs, nvar]
      - weights[nobs, nvar]

    Optional:
      - niter    : maximum number of iterations
      - nvec     : number of model vectors
      - smooth   : smoothing length scale (0 for no smoothing)
      - seed     : random seed
      - randseed : random number generator seed; None to not re-initialize

    Returns Model object
    """
    if weights is None:
        weights = np.ones(data.shape)

    assert data.shape == weights.shape

    model = Model(data, weights, nvec)
    model.solve_coeffs()

    if not silent:
        print('       iter        R2             rchi2')

    for k in range(1, niter + 1):
        model.solve_coeffs()
        model.solve_eigenvectors(*args, **kwargs)
        if not silent:
            print(f'EMPCA {k}/{niter} : {model.R2():15.8f} {model.rchi2():15.8f}')
            sys.stdout.flush()

    model.solve_coeffs()

    if not silent:
        print(f'R2: {model.R2()}')

    return model
