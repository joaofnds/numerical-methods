import numpy as np


def diagonally_dominant(a: np.ndarray) -> bool:
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        return False

    a_abs = np.abs(a)
    diag = np.diag(a_abs)
    rows_sum = np.sum(a_abs, axis=1)

    return np.all(diag > rows_sum - diag)
