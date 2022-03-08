from ..util.fitting import *
from ..util.misc import prune_data


class ContinuumModel:

    def __init__(self, data, wave_range=None):
        self.data = data.copy()
        self.data = prune_data(self.data)

        self._max_flux = self.data[:, 1].max()

        self.params = None

    def fit(self, fit_method=None, *args, **kwargs):
        y_obs = self.data[:, 1] / self._max_flux
        try:
            y_err_obs = self.data[:, 2] / self._max_flux
        except IndexError:
            y_err_obs = None

        if fit_method == 'ls':
            params, cov = least_squares(self.function, self.data[:, 0], y_obs,
                                        p0=self.params, sigma=y_err_obs,
                                        *args, **kwargs)

        self.params = params
        return params

    def predict(self, x_pred):
        return self._max_flux * self.function(x_pred, *self.params)

    def function(self):
        pass
