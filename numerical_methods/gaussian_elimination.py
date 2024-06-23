import numpy as np


def gaussian_elimination(A, b):
    Ab = np.hstack(
        (A, np.array(b, dtype=np.float64).reshape(-1, 1)),
    )
    nrows = len(b)

    # forward elimination
    for i in range(nrows):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i  # find row with max value in the i-th column
        Ab[[i, max_row]] = Ab[[max_row, i]]

        for k in range(i + 1, nrows):
            factor = Ab[k, i] / Ab[i, i]
            Ab[k, :] -= factor * Ab[i, :]

    # back substitution
    x = np.zeros(nrows)
    for i in range(nrows - 1, -1, -1):
        x[i] = (Ab[i, nrows] - np.dot(Ab[i, i + 1 : nrows], x[i + 1 : nrows])) / Ab[i, i]

    return x.tolist()
