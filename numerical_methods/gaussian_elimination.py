import numpy as np


def gaussian_elimination(A: np.ndarray, b: np.ndarray):
    Ab = np.column_stack((A, b))
    nrows = len(b)

    # forward elimination
    for row in range(nrows):
        max_row = abs(Ab[row:, row]).argmax() + row
        Ab[[row, max_row]] = Ab[[max_row, row]]

        for k in range(row + 1, nrows):
            factor = Ab[k, row] / Ab[row, row]
            Ab[k] -= factor * Ab[row]

    # back substitution
    x = np.zeros(nrows)
    for row in range(nrows - 1, -1, -1):
        x[row] = (Ab[row, nrows] - np.dot(Ab[row, row + 1 : nrows], x[row + 1 : nrows])) / Ab[row, row]

    return x
