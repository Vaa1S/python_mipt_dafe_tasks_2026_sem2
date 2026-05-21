import numpy as np


class ShapeMismatchError(Exception):
    pass


def can_satisfy_demand(
    costs: np.ndarray,
    resource_amounts: np.ndarray,
    demand_expected: np.ndarray,
) -> bool:
    if costs.ndim != 2:
        raise ShapeMismatchError

    m, n = costs.shape

    if resource_amounts.shape != (m,):
        raise ShapeMismatchError

    if demand_expected.shape != (n,):
        raise ShapeMismatchError

    resources_required = costs @ demand_expected

    return bool(np.all(resources_required <= resource_amounts))
