import numpy as np


def lu_decomposition(A: np.ndarray):
    L = np.eye(len(A))
    U = np.zeros_like(L)

    for k in range(len(A)):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]

        for i in range(k + 1, len(A)):
            if U[k, k] == 0:
                raise ZeroDivisionError()

            L[i, k] = (A[i, k] - L[i, :k] @ U[:k, k]) / U[k, k]

    return L, U


def solve_lu(A: np.ndarray, b: np.ndarray):
    L, U = lu_decomposition(A)

    # forward substitution (Ly = b)
    y = np.zeros(len(L))
    for i in range(len(y)):
        y[i] = b[i] - L[i, :i] @ y[:i]

    # backward substitution (Ux = y)
    x = np.zeros(len(U))
    for i in range(len(x) - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]

    return x
