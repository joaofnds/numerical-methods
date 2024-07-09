import numpy as np

from lib.diagonally_dominant import diagonally_dominant
from numerical_methods.gauss_trail import GaussTrail


def gauss_seidel(A: np.ndarray, b: np.ndarray, tol: float = None, max_iter: int = 1000):
    # assert np.abs(np.max(np.linalg.eigvals(A))) < 1, "matrix A must be convergent"
    assert diagonally_dominant(A), "matrix A must be diagonally dominant"

    n = len(A)
    x = np.zeros_like(b)
    trail = GaussTrail()
    trail.record(x)

    for _ in range(max_iter):
        x_new = np.zeros_like(b)

        for i in range(n):
            s1 = A[i, :i] @ x_new[:i]
            s2 = A[i, i + 1 :] @ x[i + 1 :]
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if tol is not None and np.max(np.abs(x_new - x)) / np.max(np.abs(x_new)) < tol:
            break

        x = x_new
        trail.record(x)

    return trail
