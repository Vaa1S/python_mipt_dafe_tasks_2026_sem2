import numpy as np


class ShapeMismatchError(Exception):
    pass


def get_projections_components(
    matrix: np.ndarray,
    vector: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ShapeMismatchError

    if matrix.shape[1] != vector.shape[0]:
        raise ShapeMismatchError

    n = matrix.shape[0]

    if np.linalg.matrix_rank(matrix) != n:
        return None, None

    scalar_products = matrix @ vector
    squared_norms = np.sum(matrix ** 2, axis=1)
    coefficients = scalar_products / squared_norms

    projections = coefficients[:, np.newaxis] * matrix
    components = vector - projections

    return projections, components