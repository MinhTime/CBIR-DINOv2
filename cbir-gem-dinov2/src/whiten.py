"""Learned whitening Lw (Radenović et al.), ported from cirtorch/utils/whiten.py."""
import numpy as np


def cholesky(S):
    alpha = 0
    while True:
        try:
            return np.linalg.cholesky(S + alpha * np.eye(*S.shape))
        except np.linalg.LinAlgError:
            alpha = 1e-10 if alpha == 0 else alpha * 10


def whitenlearn(X, qidxs, pidxs):
    """X: D x N descriptors; (qidxs[i], pidxs[i]) are matching pairs. Returns (m, P)."""
    m = X[:, qidxs].mean(axis=1, keepdims=True)
    df = X[:, qidxs] - X[:, pidxs]
    S = df @ df.T / df.shape[1]
    P = np.linalg.inv(cholesky(S))
    df = P @ (X - m)
    eigval, eigvec = np.linalg.eigh(df @ df.T)
    order = eigval.argsort()[::-1]
    return m, eigvec[:, order].T @ P


def whitenapply(X, m, P):
    X = P @ (X - m)
    return X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-6)
