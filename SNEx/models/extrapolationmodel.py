from ..util.fitting import *
from ..util.misc import prune_data
from ..util.misc import get_normalization


class ExtrapolationModel:

    def __init__(self, data, fit_range=None, *args, **kwargs):
        self.data = data.copy()
        self.data = prune_data(self.data, fit_range)

        self._norm = get_normalization(self.data[:, 1], *args, **kwargs)

        self._params = None

    def fit(self, fit_method=None, *args, **kwargs):
        y_obs = self.data[:, 1] / self._norm
        try:
            y_err_obs = self.data[:, 2] / self._norm
        except IndexError:
            y_err_obs = None

        if fit_method is None:
            fit_method = 'ls'

        if fit_method == 'ls':
            params, cov = least_squares(self.function, self.data[:, 0], y_obs,
                                        p0=self._params, sigma=y_err_obs,
                                        *args, **kwargs)

        self._params = params
        return params

    def predict(self, x_pred, *args, **kwargs):
        return self._norm * self.function(x_pred, *self._params)

    def function(self):
        pass
