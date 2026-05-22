import numpy as np


def get_extremum_indices(
    ordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if ordinates.size < 3:
        raise ValueError

    middle = ordinates[1:-1]
    left = ordinates[:-2]
    right = ordinates[2:]

    minima_mask = (middle < left) & (middle < right)
    maxima_mask = (middle > left) & (middle > right)

    minima_indices = np.where(minima_mask)[0] + 1
    maxima_indices = np.where(maxima_mask)[0] + 1

    return minima_indices, maxima_indices
