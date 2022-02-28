from .models import Wien, Planck
from .util.filter import monotonic
from .util.fitting import fit_methods
from .util.io import read_spectrum
from .util.misc import check_method

_NIR_methods = ('planck', 'wien')
_UV_method = ('',)


# TODO: include exctinction in SNEx init/read_spectrum


class SNEx:

    def __init__(self, data, z=None, wave_range=None, *args, **kwargs):
        if isinstance(data, str):
            self.data = read_spectrum(data, z, wave_range, **kwargs)
        else:
            self.data = data

    def predict(self, x_pred, regime='NIR', fit_range=None, extrap_method=None,
                fit_method=None, filter_method=None, *args, **kwargs):
        fit_method = check_method(fit_method, fit_methods)

        if regime.lower() == 'nir':
            return self._predict_NIR(x_pred, fit_range, fit_method,
                                     extrap_method, filter_method,
                                     *args, **kwargs)
        elif regime.lower() == 'uv':
            return self._predict_UV(x_pred, fit_range, fit_method,
                                    extrap_method, filter_method,
                                    *args, **kwargs)
        else:
            raise RuntimeError('Unknown wavelength regime.')

    def _filter(self, data, filter_method):
        if filter_method is None:
            return data

        if filter_method.lower() == 'monotonic':
            filtered_data = monotonic(self.data)

        return filtered_data

    def _predict_NIR(self, x_pred, fit_range=None, fit_method=None,
                     extrap_method=None, filter_method=None, *args, **kwargs):
        extrap_method = check_method(extrap_method, _NIR_methods)

        data = self._filter(self.data, filter_method)

        if fit_range is None:
            fit_range = (6200., 12000.)

        if extrap_method == 'wien':
            print('Predicting with Wien model...')
            model = Wien(data, wave_range=fit_range)
        elif extrap_method == 'planck':
            print('Predicting with Planck model...')
            model = Planck(data, wave_range=fit_range)

        model.fit(fit_method=fit_method, *args, **kwargs)
        print(f'   {model}')
        return model.predict(x_pred)

    def _predict_UV(self, x_pred, fit_range=None, fit_method=None,
                    extrap_method=None, filter_method=None, *args, **kwargs):
        extrap_method = check_method(extrap_method, _UV_method)
