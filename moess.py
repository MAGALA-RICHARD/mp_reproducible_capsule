import numpy as np


def moess(F, w=None, eps=1e-12):
    """
    Compute MOESS across candidate solutions.

    Parameters
    ----------
    F : np.ndarray, shape (n_samples, n_objectives)
        Matrix of objective values. Each row is one candidate, each column an objective.
    w : array-like or None
        Weights for each objective. If None, equal weights are used.
    eps : float
        Small tolerance to avoid division by zero when max == min.

    Returns
    -------
    moess_min : float
        Minimum MOESS value.
    idx : int
        Index of row achieving the min.
    row_values : np.ndarray
        The objective values at the argmin.
    all_scores : np.ndarray, shape (n_samples, )
        MOESS value for every candidate row.
    """
    F = np.asarray(F, dtype=float)
    n_samples, n_obj = F.shape

    if w is None:
        w = np.ones(n_obj) / n_obj
    else:
        if sum(w) > 1 + eps:
            raise ValueError("weights must sum to 1")
        w = np.asarray(w, dtype=float)
        if w.size != n_obj:
            raise ValueError("Length of weights must match number of objectives")

    # Normalize weights
    w = w / w.sum()

    # Min/max per objective
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    ranges = np.maximum(f_max - f_min, eps)

    # Apply MOESS formula for each row
    scores = ((f_max - F) / ranges) @ w

    idx = int(np.argmin(scores))
    return float(scores[idx]), idx, F[idx], scores


# ---- Example ----
if __name__ == "__main__":
    # 5 candidates × 3 objectives
    F = np.array([
        [10, 500, 0.2],
        [8, 1000, 0.5],
        [12, 1500, 0.4],
        [6, 800, 0.3],
        [9, 1200, 0.6],
    ])

    moess_val, idx, combo, all_scores = moess(F, w=[0.5, 0.3, 0.2])
    print("Best MOESS value:", moess_val)
    print("At row index    :", idx)
    print("Row values      :", combo)
    print("All scores      :", all_scores)
