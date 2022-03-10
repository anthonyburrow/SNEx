from .models import Wien, Planck, PCA
from .util.filter import monotonic
from .util.fitting import fit_methods
from .util.io import read_spectrum
from .util.misc import check_method

_NIR_methods = ('planck', 'wien', 'pca')
_UV_method = ('',)


# TODO: include exctinction in SNEx init/read_spectrum


class SNEx:

    def __init__(self, data, z=None, wave_range=None, *args, **kwargs):
        if isinstance(data, str):
            self.data = read_spectrum(data, z, wave_range, **kwargs)
        else:
            self.data = data

    def predict(self, regime='nir', *args, **kwargs):
        if regime.lower() == 'nir':
            return self._predict_NIR(*args, **kwargs)
        elif regime.lower() == 'uv':
            return self._predict_UV(*args, **kwargs)
        else:
            raise RuntimeError('Unknown wavelength regime.')

    def _predict_NIR(self, extrap_method=None, *args, **kwargs):
        extrap_method = check_method(extrap_method, _NIR_methods)

        data = self._filter(self.data, *args, **kwargs)

        if extrap_method == 'wien':
            print('Predicting with Wien model...')
            model = Wien(data, *args, **kwargs)
        elif extrap_method == 'planck':
            print('Predicting with Planck model...')
            model = Planck(data, *args, **kwargs)
        elif extrap_method == 'pca':
            print('Predicting with NIR PCA model...')
            model = PCA('nir', data=data, *args, **kwargs)

        # TODO: Add extra attributes from specific model to SNEx model here

        model.fit(*args, **kwargs)
        print(f'   {model}')
        return model.predict(*args, **kwargs)

    def _predict_UV(self, x_pred=None, fit_range=None, fit_method=None,
                    extrap_method=None, filter_method=None, *args, **kwargs):
        extrap_method = check_method(extrap_method, _UV_method)

    def _filter(self, data, filter_method=None, *args, **kwargs):
        if filter_method is None:
            return data

        if filter_method.lower() == 'monotonic':
            filtered_data = monotonic(self.data)

        return filtered_data
