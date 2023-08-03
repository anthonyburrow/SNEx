from ..util.fitting import *
from ..util.misc import prune_data, get_normalization, between_mask


normalization_region = (5000., 8400.)
norm_method = 'mean'


class ExtrapolationModel:

    def __init__(self, data, fit_range=None, *args, **kwargs):
        # Separates SNEx.data (original, preprocessed) from
        # ExtrapolationModel.data (normalized, pruned)
        self.data = data.copy()

        # Get normalization before pruning
        # TODO: WARN IF NO DATA IN normalization_region
        norm_mask = between_mask(self.data[:, 0], normalization_region)
        self._norm = get_normalization(self.data[norm_mask, 1], norm_method)
        self.data[:, 1] /= self._norm
        try:
            self.data[:, 2] /= self._norm
        except IndexError:
            pass

        self.data = prune_data(self.data, fit_range)

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
