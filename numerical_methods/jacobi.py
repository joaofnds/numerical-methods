import numpy as np

from lib.diagonally_dominant import diagonally_dominant
from numerical_methods.gauss_trail import GaussTrail


def jacobi(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tol: float = None, max_iter: int = 1000):
    # assert np.abs(np.max(np.linalg.eigvals(A))) < 1, "matrix A must be convergent"
    assert diagonally_dominant(A), "matrix A must be diagonally dominant"

    L = np.tril(A, -1)
    D = np.diag(A.diagonal())
    D_inv = np.linalg.inv(D)
    U = np.triu(A, 1)

    trail = GaussTrail()

    x = x0.copy()
    trail.record(x)

    for _ in range(max_iter):
        x_new = -D_inv @ ((U + L) @ x) + D_inv @ b

        if tol is not None and np.max(np.abs(x_new - x)) / np.max(np.abs(x_new)) < tol:
            break

        x = x_new
        trail.record(x)

    return trail
