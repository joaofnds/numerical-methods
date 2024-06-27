import numpy as np


def cholesky_decomposition(A: np.ndarray):
    assert np.allclose(A, A.T)
    assert np.all(np.linalg.eigvals(A) > 0)

    L = np.zeros_like(A, dtype=np.float64)

    for i in range(len(A)):
        L[i, i] = np.sqrt(A[i, i] - L[i, :] @ L[i, :])

        for j in range(i + 1, len(A)):
            L[j, i] = (A[j, i] - L[i, :] @ L[j, :]) / L[i, i]

    return L


def solve_cholesky(A: np.ndarray, b: np.ndarray):
    L = cholesky_decomposition(A)

    # forward substitution (Ly = b)
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(len(A)):
        y[i] = (b[i] - L[i] @ y) / L[i, i]

    # backward substitution (L^T x = y)
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(len(A) - 1, -1, -1):
        x[i] = (y[i] - L[i + 1 :, i] @ x[i + 1 :]) / L[i, i]

    return x
