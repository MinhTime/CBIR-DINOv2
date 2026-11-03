"""alpha-weighted query expansion (Radenović et al., TPAMI 2018, Sec. 4)."""
import numpy as np


def aqe(X, Q, n=2, alpha=3):
    """q' = l2n(q + sum_{i<=n} sim_i^alpha * x_i). X: D x Ndb, Q: D x Nq (L2-normalized)."""
    S = X.T @ Q
    top = np.argsort(-S, axis=0)[:n]
    Qn = Q.copy()
    for j in range(Q.shape[1]):
        w = np.clip(S[top[:, j], j], 0, None) ** alpha
        Qn[:, j] = Q[:, j] + (X[:, top[:, j]] * w).sum(1)
    return Qn / (np.linalg.norm(Qn, axis=0, keepdims=True) + 1e-6)
