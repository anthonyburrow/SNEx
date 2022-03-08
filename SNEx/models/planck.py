from numpy import exp

from .continuummodel import ContinuumModel
from ..util.constants import c, h, k_B


# Temperature, arbitrary constant
_default_params = (10000., 1e0)


class Planck(ContinuumModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = _default_params

    def function(self, wave, T, A):
        const = h * c * 1e8 / (k_B * T)
        return 1e18 * A * wave**(-5) / (exp(const / wave) - 1.)

    def __str__(self):
        return f'T = {self.params[0]:.3f} K'
