def rank_k_approx(A, k):
    """Return the rank-k SVD approximation of A."""
    import numpy as np

    A = np.asarray(A, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    return (U[:, :k] * s[:k]) @ Vt[:k, :]
