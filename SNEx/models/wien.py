from numpy import exp

from .extrapolationmodel import ExtrapolationModel
from ..util.constants import c, h, k_B


# Temperature, arbitrary constant
_default_params = (10000., 1e0)


class Wien(ExtrapolationModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._params = _default_params

    def function(self, wave, T, A):
        const = h * c * 1e8 / (k_B * T)
        return 1e18 * A * wave**(-5) * exp(-const / wave)

    def __str__(self):
        return f'T = {self._params[0]:.3f} K'
