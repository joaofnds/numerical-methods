import numpy as np


def partial_pivoting(A: np.ndarray, b: np.ndarray):
    n = A.shape[0]
    P = np.eye(n)

    for k in range(n):
        max_index = np.argmax(np.abs(A[k:, k])) + k
        if max_index != k:
            A[[k, max_index]] = A[[max_index, k]]
            P[[k, max_index]] = P[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]

    return P, A, b
