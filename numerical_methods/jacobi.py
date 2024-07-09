import numpy as np


def jacobi(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tol: float = None, max_iter: int = 1000):
    assert np.all(np.linalg.eigvals(A) > 0)

    L = np.tril(A, -1)
    D = np.diag(A.diagonal())
    D_inv = np.linalg.inv(D)
    U = np.triu(A, 1)

    x = x0.copy()

    for _ in range(max_iter):
        x_new = -D_inv @ ((U + L) @ x) + D_inv @ b

        if tol is not None and np.max(np.abs(x_new - x)) / np.max(np.abs(x_new)) < tol:
            break

        x = x_new

    return x
