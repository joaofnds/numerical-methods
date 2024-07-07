import numpy as np


def lup_decomposition(A: np.ndarray):
    n = A.shape[0]

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        # partial pivoting
        max_row = np.argmax(np.abs(U[k:, k])) + k
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            L[[k, max_row], :k] = L[[max_row, k], :k]
            P[[k, max_row]] = P[[max_row, k]]

        # elimination
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return L, U, P


def solve_lup(A: np.ndarray, b: np.ndarray):
    L, U, P = lup_decomposition(A)

    b = P @ b
    n = A.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]

    return x
