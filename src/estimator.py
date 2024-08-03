import numba
import numpy as np

@numba.njit("float64[:](float64[:, :], float64[:], int64)")
def LURE(l, q, N):
    M = len(q)
    h = np.zeros(l.shape[1], dtype=float)
    for m in range(1, M+1):
        v = 1 + (1 / ((N - m + 1) * q[m-1]) - 1) * (N - M) / (N - m + 1e-8)
        h = h + v * l[m-1] / M
    return h

def WAVG(l, q, N):
    L = np.mean(l / q[:, np.newaxis], axis=0)
    W = np.mean(1 / q)
    return L / W

def AVG(l, q, N):
    L = np.mean(l, axis=0)
    return L
