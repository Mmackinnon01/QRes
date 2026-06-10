import numpy as np


def tsvd_pinv(A, r=None, tol=None, energy=None):
    """
    Truncated-SVD pseudoinverse of A.

    Choose exactly one of:
      - r: keep this many largest singular values
      - tol: keep s_i >= tol
      - energy: keep smallest r such that sum_{i<=r} s_i^2 / sum s_i^2 >= energy  (e.g., 0.999)

    Returns: A_pinv, svals, r_eff, cutoff
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    if r is None and tol is None and energy is None:
        # Default Lapack-like tolerance
        tol = (
            max(A.shape) * np.finfo(A.dtype if np.iscomplexobj(A) else float).eps * s[0]
        )

    if energy is not None:
        cum = np.cumsum(s**2) / np.sum(s**2) if np.sum(s**2) > 0 else np.zeros_like(s)
        r_eff = int(np.searchsorted(cum, energy) + 1)
    elif tol is not None:
        r_eff = int(np.sum(s >= tol))
    else:
        r_eff = int(min(len(s), r))

    r_eff = max(0, min(r_eff, len(s)))
    inv_s = np.zeros_like(s)
    inv_s[:r_eff] = 1.0 / s[:r_eff]
    A_pinv = (Vh.T * inv_s) @ U.T.conj()
    cutoff = s[r_eff - 1] if r_eff > 0 else 0.0
    return A_pinv, s, r_eff, cutoff


def tsvd_solve(A, b, r=None, tol=None, energy=None):
    """
    Solve min ||Ax - b|| with TSVD regularization; supports multiple RHS columns in b.
    Returns x, info dict.
    """
    A = np.hstack([A, np.ones((A.shape[0], 1))])
    A_pinv, s, r_eff, cutoff = tsvd_pinv(A, r=r, tol=tol, energy=energy)
    x = A_pinv @ b

    return x[:-1], x[-1]


def tsvd_solve_no_intercept(A, b, r=None, tol=None, energy=None):
    """
    Solve min ||Ax - b|| with TSVD regularization; supports multiple RHS columns in b.
    Returns x, info dict.
    """
    A_pinv, s, r_eff, cutoff = tsvd_pinv(A, r=r, tol=tol, energy=energy)
    x = A_pinv @ b

    return x


class LinearRegression:

    def fit(self, x, y, fit_intercept=False, r=None):
        if fit_intercept:
            self.coef_, self.intercept_ = tsvd_solve(x, y, r=r)
        else:
            self.coef_, self.intercept_ = tsvd_solve_no_intercept(x, y, r=r), 0

    def predict(self, x):
        return x @ self.coef_ + self.intercept_
