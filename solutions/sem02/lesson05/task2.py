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

    projections = np.zeros((n, n), dtype=float)
    components = np.zeros((n, n), dtype=float)

    for i in range(n):
        basis_vector = matrix[i]
        scalar = np.sum(vector * basis_vector)
        norm_squared = np.sum(basis_vector * basis_vector)
        projection = scalar / norm_squared * basis_vector
        projections[i] = projection
        components[i] = vector - projection

    return projections, components
