import numpy as np


class ShapeMismatchError(Exception):
    pass


def adaptive_filter(
    Vs: np.ndarray,
    Vj: np.ndarray,
    diag_A: np.ndarray,
) -> np.ndarray:
    if Vs.ndim != 2 or Vj.ndim != 2 or diag_A.ndim != 1:
        raise ShapeMismatchError

    if Vs.shape[0] != Vj.shape[0]:
        raise ShapeMismatchError

    if Vj.shape[1] != diag_A.shape[0]:
        raise ShapeMismatchError

    m = Vs.shape[0]
    k = Vj.shape[1]

    Vj_H = Vj.conj().T
    A = np.diag(diag_A)

    inner = np.eye(k) + Vj_H @ Vj @ A
    inner_inv = np.linalg.inv(inner)

    R_inv = np.eye(m) - Vj @ inner_inv @ Vj_H

    return R_inv @ Vs
