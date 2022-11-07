from .models import Wien, Planck, PCA
from .util.filter import monotonic
from .util.io import read_spectrum

_default_NIR = 'pca'
_default_UV = ''


class SNEx:

    def __init__(self, data, *args, **kwargs):
        if isinstance(data, str):
            self.data = read_spectrum(data, *args, **kwargs)
        else:
            self.data = data

    def predict(self, regime='nir', *args, **kwargs):
        """Make a prediction to extrapolate spectra in the NIR or UV.

        Parameters
        ----------
        x_pred : numpy.ndarray, optional
            Needed for Wien and Planck models. These are the wavelengths at
            which the model is predicting. Not used in PCA as the PCA model
            requires specific wavelength points for prediction.
        regime : str, optional
            Wavelength regime ('uv' or 'nir'). Defaults to 'nir'.
        extrap_method : str, optional
            Type of model to use. Allowed NIR models are 'pca', 'wien',
            'planck'. Allow UV models are 'pca'. The PCA model is default for
            each case.
        filter_method : str, optional
            Perform a filtering method before training the model to limit the
            effect of large absorption features. Allow methods are 'monotonic'
            and 'boxcar'. No filter is applied by default.
        fit_range : tuple, optional
            2-tuple of floats indicating the minimum and maximum wavelengths
            with which to train the extrapolation model.
        fit_method : str, optional
            For some models (Wien, Planck) a fitting technique is applied.
            Allowed methods are 'ls' (Least-Squares, default).
        bounds : tuple, optional
            Bounds on the model parameters for 'ls' fitting. This should be the
            same format as the bounds parameter in scipy.optimize.curve_fit().
        time : float, optional
            (PCA) The time of observation in days past max B light. By default
            this assumes the spectrum is at/near maximum light.
        n_components : int, optional
            (PCA) The number of PC eigenvectors to predict with.

        Returns
        -------
        numpy.ndarray
            The predicted values at wavelengths given by x_pred (if used).
        numpy.ndarray
            In some models (PCA) the associated wavelengths are returned in
            addition.
        """
        regime = regime.lower()

        data = self._filter(self.data, *args, **kwargs)

        if regime == 'nir':
            model = self._get_NIR_model(data, *args, **kwargs)
        elif regime == 'uv':
            model = self._get_UV_model(data, *args, **kwargs)
        else:
            raise RuntimeError('Unknown wavelength regime.')

        # TODO: Add extra attributes from specific model to SNEx model here

        model.fit(*args, **kwargs)
        # print(f'   {model}')
        return model.predict(*args, **kwargs)

    def _get_NIR_model(self, data, extrap_method=None, *args, **kwargs):
        if extrap_method is None:
            extrap_method = _default_NIR
        extrap_method = extrap_method.lower()

        if extrap_method == 'wien':
            print('Predicting with Wien model...')
            model = Wien(data, *args, **kwargs)
        elif extrap_method == 'planck':
            # print('Predicting with Planck model...')
            model = Planck(data, *args, **kwargs)
        elif extrap_method == 'pca':
            print('Predicting with NIR PCA model...')
            model = PCA(data=data, regime='nir', *args, **kwargs)

        return model

    def _get_UV_model(self, data, extrap_method=None, *args, **kwargs):
        if extrap_method is None:
            extrap_method = _default_UV
        extrap_method = extrap_method.lower()

        if extrap_method == 'pca':
            print('Predicting with NIR PCA model...')
            model = PCA('uv', data=data, *args, **kwargs)

        return model

    def _filter(self, data, filter_method=None, *args, **kwargs):
        if filter_method is None:
            return data

        if filter_method.lower() == 'monotonic':
            filtered_data = monotonic(self.data)

        return filtered_data
