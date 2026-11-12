def rank_k_approx(A, k):
    """Return the rank-k SVD approximation of A."""
    import numpy as np

    A = np.asarray(A, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    return (U[:, :k] * s[:k]) @ Vt[:k, :]

def storage(m, n, k):
    """Number of values stored in rank-k SVD factors."""
    return k * (m + n + 1)


def compression_ratio(m, n, k):
    """Original value count divided by SVD factor value count."""
    return (m * n) / storage(m, n, k)


def rel_error(A, B):
    """Relative Frobenius error when B approximates A."""
    return np.linalg.norm(A - B) / np.linalg.norm(A)


def predicted_error(s, k):
    """Relative rank-k error from the singular values."""
    return np.sqrt(
        np.sum(s[k:] ** 2) / np.sum(s ** 2)
    )
