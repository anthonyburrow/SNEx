import numpy as np
from spextractor.physics.lines import sn_lines


line_ids = {
    'Ca II': (
         8_498.,
         8_542.,
         8_662.,
    ),
    'Mg II': (
         9_227.,
        10_092.,
        10_927.,
        16_787.,
        21_569.,
    ),
    'Si III': (
        12_523.,
        12_601.,
        13_395.,
        13_497.,
        13_644.,
    )
}


def get_line_location(data: np.ndarray, feature: str,
                      lower_bounds: tuple = None, upper_bounds: tuple = None):
    feat = sn_lines[feature]

    if lower_bounds is None:
        lower_bounds = feat['lo_range']
    if upper_bounds is None:
        upper_bounds = feat['hi_range']

    wave = data[:, 0]
    flux = data[:, 1]

    lower_mask = (lower_bounds[0] <= wave) & (wave <= lower_bounds[1])
    upper_mask = (upper_bounds[0] <= wave) & (wave <= upper_bounds[1])

    # Feature is outside spectrum
    if not (np.any(lower_mask) and np.any(upper_mask)):
        print(f'Feature around [{lower_bounds[0]}, {upper_bounds[1]}] outside spectrum')
        return

    lo_max_ind = flux[lower_mask].argmax()
    hi_max_ind = flux[upper_mask].argmax()

    lo_max_wave = wave[lower_mask][lo_max_ind]
    hi_max_wave = wave[upper_mask][hi_max_ind]

    feature_mask = (lo_max_wave <= wave) & (wave <= hi_max_wave)
    flux_minimum_ind = flux[feature_mask].argmin()

    # There is no local minimum
    if flux_minimum_ind == 0 or flux_minimum_ind == len(flux[feature_mask]) - 1:
        print(f'No local minimum within range [{lo_max_wave}, {hi_max_wave}]')
        return

    wave_of_minimum = wave[feature_mask][flux_minimum_ind]

    return wave_of_minimum
