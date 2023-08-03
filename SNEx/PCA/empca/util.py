import numpy as np


def norm(x):
    '''
    Faster alternative to 'np.linalg.norm(x)'
    '''
    return np.sqrt(x @ x)


def norm2D(x):
    '''
    Faster alternative to 'np.linalg.norm(x, axis=1)'
    '''
    return np.sqrt((x * x).sum(axis=1))


def random_orthonormal(nvec, nvar, seed=None, *args, **kwargs):
    """
    Return array of random orthonormal vectors A[nvec, nvar]

    Doesn't protect against rare duplicate vectors leading to 0s
    """

    if seed is not None:
        np.random.seed(seed)

    A = np.random.normal(size=(nvec, nvar))

    # Normalize each eigenvector
    A = (A.T / norm2D(A)).T

    # Gram-Schmidt method
    for i in range(1, nvec):
        for j in range(0, i):
            A[i] -= (A[j] @ A[i]) * A[j]
            A[i] /= norm(A[i])

    return A
