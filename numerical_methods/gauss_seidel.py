import numpy as np


def gauss_seidel(A: np.ndarray, b: np.ndarray, tol: float = None, max_iter: int = 1000):
    n = len(A)
    x = np.zeros_like(b)

    for _ in range(max_iter):
        x_new = np.zeros_like(b)

        for i in range(n):
            s1 = A[i, :i] @ x_new[:i]
            s2 = A[i, i + 1 :] @ x[i + 1 :]
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if tol is not None and np.max(np.abs(x_new - x)) / np.max(np.abs(x_new)) < tol:
            break

        x = x_new

    return x
