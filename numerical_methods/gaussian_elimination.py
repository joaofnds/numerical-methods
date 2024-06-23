import numpy as np


def gaussian_elimination(A: np.ndarray, b: np.ndarray, complete_pivoting=False):
    Ab = np.column_stack((A, b))
    nrows, ncols = Ab.shape
    order = np.arange(nrows)

    # forward elimination
    for row in range(nrows):
        if complete_pivoting:
            max_idx = np.unravel_index(np.abs(Ab[row:, row : ncols - 1]).argmax(), Ab[row:, row : ncols - 1].shape)
            max_row, max_col = max_idx[0] + row, max_idx[1] + row

            Ab[[row, max_row]] = Ab[[max_row, row]]
            Ab[:, [row, max_col]] = Ab[:, [max_col, row]]
            order[[row, max_col]] = order[[max_col, row]]

        else:
            max_row = abs(Ab[row:, row]).argmax() + row
            Ab[[row, max_row]] = Ab[[max_row, row]]

        for k in range(row + 1, nrows):
            factor = Ab[k, row] / Ab[row, row]
            Ab[k] -= factor * Ab[row]

    # back substitution
    x = np.zeros(nrows)
    for row in range(nrows - 1, -1, -1):
        x[row] = (Ab[row, nrows] - np.dot(Ab[row, row + 1 : nrows], x[row + 1 : nrows])) / Ab[row, row]

    return x[order]
