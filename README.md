# SNEx
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SNEx (SuperNova Extrapolation) is a tool that makes use of Principal Component
Analysis (PCA) performed on a data set of Carnegie Supernova Project (CSP) I &
II optical and near-infrared (NIR) spectra to extrapolate optical spectra of
Type Ia SNe into the NIR, up to about 2.3 microns.

A summarization of this method is the following: PCA is performed on subsets of
CSP spectra in different time windows to provide eigenvectors (principal
components, or PCs) that describe the variation of the subset at different
times. The user may then project a separate optical Type Ia spectrum onto the
optical region of the PCs to get a set of projections. These projections are
then combined with the full PCs that extend into the NIR to form a linear
combination, which is the resulting prediction or extrapolation. For more
detail, this method and results are discussed at length in the upcoming
publication by Burrow et al. (submitted).

This software is also intended to be a generalized extrapolator for spectra.
There are other options available aside from using PCA as described in Burrow
et al. (submitted), for example a Planck function. This generalization will be
more documented and user-friendly in future updates. As more data is available,
the PCA method may be extended into the UV as well. The code here is
partially set up to allow such extensions, but is not complete or useable yet
as no models exist. This is subject to more future work.

## Use

A `setup.py` will be added soon for easy installation of this package.
Otherwise, add the SNEx source code to your Python path, and import the SNEx
module that way.

For a more detailed general use case, see the example scripts provided in
`examples/`. Essentially, the basic use of SNEx is the following:

First, instantiate SNEx with a spectrum. SNEx uses Spextractor
(`https://github.com/anthonyburrow/spextractor`) to preprocess the spectrum.
The spectrum must be preprocessed (put into rest frame, corrected for
extinction, etc.) to be fully compatible with the model eigenvectors. See
Burrow et al. (submitted) for the detail on how these eigenvectors are created.
```python
fn = 'my_spectrum.dat'

read_params = {
    'z': 0.001,
#    ...
}

model = SNEx(fn, **read_params)
```

Next, the prediction is made using the spectrum and eigenvectors that are
loaded. This is simply done with the `.predict()` method. This method takes
several different parameters, which are described in the following Parameters
section. A tuple of arrays is returned, which include `y_pca` (the predicted
flux values), `y_err_pca` (the calculated uncertainty in `y_pca`), and `x_pca`
(the wavelength values that the eigenvectors, and therefore the `y_pca` values,
use as a basis.)
```python
params = {
    'time': 0.,
    'extrap_method': 'pca',
    'fit_range': (5000., 8400.),
    'predict_range': (8400., 23000.),
    'n_components': 10,
}
y_pca, y_err_pca, x_pca = model.predict(**params)
```

## Parameters

The prediction has several arguments available for the PCA model. They key
arguments for extrapolation with PCA are:
```
time : float, optional
    (PCA) The time of observation in days past max B light. By default
    this assumes the spectrum is at/near maximum light.
extrap_method : str, optional
    Type of model to use. Allowed NIR models are 'pca', 'wien',
    'planck'. Allow UV models are 'pca'. The PCA model is default for
    each case.
fit_range : tuple, optional
    2-tuple of floats indicating the minimum and maximum wavelengths
    with which to train the extrapolation model.
predict_range : tuple, optional
    (PCA) 2-tuple of floats indicating the minimum and maximum wavelengths
    of the region to be extrapolated.
n_components : int, optional
    (PCA) The number of PC eigenvectors to predict with.
```

## Citation

This software is described in detail in the upcoming publication by Burrow et
al. (submitted), which can be cited in discussion of this technique. The latest
release is also given a DOI, which may be used to reference use of this
software.
