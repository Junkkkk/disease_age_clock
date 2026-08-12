"""
optmed.py
---------
Companion code for:
    "Learning the Optimal Composite Mediator, with Application to
     Proteomics Aging Clocks"

Provides closed-form solvers for finding the composite mediator w* that
maximises the Baron-Kenny indirect effect (MaxIE) or the mediation index
f* = cor(Xw, A) * cor(Zw, Y) (MaxCor), along with OLS baselines, simulation
utilities, and functions that reproduce every table and figure in the paper.

Notation: VZ = X'X - a a' / ||A||^2  (Gram matrix of A-residualised design);
          VX = X'X  (Gram matrix of X);  a = X'A,  z = X'Y - (A'Y/||A||^2)*a.

Sections
--------
1. Helpers             (_centre, _canonical_sign, _sufficient_stats,
                        _grid_bisect, _reconstruct)
2. Solvers             (solve_maxie, cosine_test, solve_maxcor)
3. OLS baselines       (solve_reg_YX, solve_reg_AX)
4. Numerical baselines (solve_numerical_h, solve_numerical_fstar)
5. DM baseline         (solve_dm)
6. Mediation stats     (mediation_stats)
7. Data generation     (generate_data)
8. Worked example      (worked_example)
9. Paper outputs       (table_obj_h, table_obj_f, table_time,
                        figure_null_dist, table_inference)
10. Master runner      (run_all)  -- call this to reproduce all outputs

Dependencies: numpy, scipy, pandas
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# =============================================================================
# 1. Helpers
# =============================================================================

def _centre(X, A, Y):
    """Coerce inputs to float64 arrays, then column-centre X and centre A, Y.
    Accepts pandas DataFrames/Series, integer arrays, or any array-like type."""
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    return X - X.mean(0), A - A.mean(), Y - Y.mean()


def _canonical_sign(w, X, A):
    """Flip w so that cor(X @ w, A) >= 0.

    Since the objective is invariant to w -> -w (both correlations flip
    simultaneously), we fix the sign convention by requiring the composite
    to be non-negatively associated with the treatment.  The sign of the
    returned objective then carries unambiguous biological meaning: positive
    indicates consistent mediation, negative indicates suppression.
    """
    w = np.asarray(w, dtype=float).ravel()
    if np.linalg.norm(w) > 1e-15 and np.dot(X @ w, A) < 0:
        w = -w
    return w


def _sufficient_stats(X, A, Y):
    """Compute a = X'A and z = X'Y - (A'Y / ||A||^2) * a.
    Assumes X, A, Y are already centred."""
    a  = X.T @ A
    AA = float(A @ A)
    if AA < 1e-15:
        raise ValueError(
            "_sufficient_stats: A has (near-)zero variance after centring. "
            "Treatment/exposure must not be constant.")
    z  = X.T @ Y - (A @ Y / AA) * a
    return a, z, AA


def _grid_bisect(c, s, k, n_grid=200, n_bisect=60):
    """Maximise g(theta) = cos(t)*(c*cos(t)+s*sin(t)) / sqrt(1-k^2*cos^2(t)).

    Uses a coarse grid of n_grid equally spaced points to locate the basin
    of the maximum, then refines with n_bisect steps of bisection.
    """
    if s < 1e-12:
        return 0.0
    thetas     = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    ct, st     = np.cos(thetas), np.sin(thetas)
    vals       = ct * (c * ct + s * st) / np.sqrt(
                     np.maximum(1.0 - k**2 * ct**2, 1e-15))
    theta_best = thetas[np.argmax(vals)]
    lo = theta_best - np.pi / n_grid
    hi = theta_best + np.pi / n_grid
    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        def g(t):
            c2 = np.cos(t)
            return c2 * (c * c2 + s * np.sin(t)) / np.sqrt(
                       max(1.0 - k**2 * c2**2, 1e-15))
        lo, hi = (mid, hi) if g(mid + 1e-10) > g(mid - 1e-10) else (lo, mid)
    return (lo + hi) / 2.0


def _reconstruct(a, z, AA, p_vec, q_vec):
    """Back-transform from whitened space to w* (used by MaxCor)."""
    s_p = np.sqrt(max(a @ p_vec, 1e-15))
    k      = s_p / np.sqrt(AA)
    c      = (z @ p_vec) / s_p
    s      = np.sqrt(max(z @ q_vec - c**2, 0.0))
    theta  = _grid_bisect(c, s, k)
    ct, st = np.cos(theta), np.sin(theta)
    if s < 1e-12:
        return p_vec / s_p
    return (ct / s_p) * p_vec + (st / s) * (
               q_vec - (c / s_p) * p_vec)


# =============================================================================
# 2. Solvers
# =============================================================================

def solve_maxie(X, A, Y, Sigma_X=None):
    """MaxIE: closed-form maximiser of h(w) = alpha_hat(w) * beta_hat(w).

    Automatically selects the appropriate implementation based on dimensions:

    **Primal** (n > p): whitens w.r.t. V_Z = X'X - aa'/||A||^2 (p x p) via
    Cholesky. Cost: O(np) sufficient statistics + O(p^3/3) Cholesky.

    **Dual** (p >= n): operates via the kernel K_Z = Z Z' (n x n) where
    Z = Q_A X. Eigendecomposes K_Z = U diag(lambda) U', recovers the
    optimal direction in the row space of X using the restriction w = X'v
    (lossless since h depends on w only through Xw, and any w in null(X)
    leaves h unchanged). Returns w = X' v* = Z' (U diag(inv2) f_a-bisector).
    Cost: O(n^2 p) kernel formation + O(n^3) eigendecomposition.

    Parameters
    ----------
    X       : (n, p) numeric array — need not be centred
    A       : (n,)   numeric vector, treatment / exposure
    Y       : (n,)   numeric vector, outcome
    Sigma_X : (p, p) known population covariance of X (optional, primal only).

    Returns
    -------
    w_plus  : (p,) consistent composite mediator (bisector of p and q).
    w_minus : (p,) suppression weight (opposite bisector, cor(Xw,A) >= 0).
    T       : float, cosine test statistic (t(p-1) primal, t(n-2) dual).
    df      : int, degrees of freedom.
    method  : str, 'primal' or 'dual'.
    """
    X, A, Y = _centre(X, A, Y)
    n, p = X.shape
    a, z, AA = _sufficient_stats(X, A, Y)

    if n > p:
        # --- Primal: Cholesky on V_Z (p x p) ---
        if Sigma_X is not None:
            VX = np.asarray(Sigma_X, dtype=float)
            if VX.shape != (p, p):
                raise ValueError(f"Sigma_X shape {VX.shape} does not match p={p}")
            VX = VX * n
            VZ = VX - np.outer(a, a) / AA
        else:
            VZ = X.T @ X - np.outer(a, a) / AA
        reg = max(1e-10 * np.trace(VZ) / p, 1e-10)
        fct = cho_factor(VZ + reg * np.eye(p), lower=True)
        p_vec = cho_solve(fct, a)
        q_vec = cho_solve(fct, z)

    else:
        # --- Dual: eigendecompose K_Z = Z Z' (n x n) ---
        # Correct dual sufficient statistics: tilde_a = K*A, tilde_z = K*Q_A*Y
        K  = X @ X.T
        Z  = X - np.outer(A, (A @ X) / AA)
        KZ = Z @ Z.T
        eigvals, eigvecs = np.linalg.eigh(KZ)
        pos = eigvals > 1e-9 * max(float(eigvals[-1]), 1e-15)
        ev = eigvals[pos]; U = eigvecs[:, pos]
        ta = K @ A                            # tilde_a = K A
        tz = K @ Y - (A @ Y / AA) * ta       # tilde_z = K Q_A Y
        inv1 = 1.0 / ev                       # K_Z^+ via 1/lambda (not 1/lambda^2)
        # Dual path vectors: tilde_p = K_Z^+ tilde_a = U diag(1/ev) U' tilde_a
        tp = U @ (inv1 * (U.T @ ta))
        tq = U @ (inv1 * (U.T @ tz))
        # Path strengths in K_Z-metric
        s_p = np.sqrt(max(float(ta @ tp), 1e-15))
        s_q = np.sqrt(max(float(tz @ tq), 1e-15))
        cos_phi = float(ta @ tq) / (s_p * s_q)
        cos_phi = float(np.clip(cos_phi, -1.0, 1.0))
        phi = float(np.arccos(cos_phi))
        df_dual = n - 2
        cos2 = cos_phi**2
        T_dual = cos_phi * np.sqrt(df_dual / max(1.0 - cos2, 1e-15))
        # Bisector in dual space
        sp_u = tp / s_p; sq_u = tq / s_q
        if abs(np.sin(phi)) < 1e-12:
            vp = sp_u; vm = -sq_u
        else:
            perp = (sq_u - cos_phi * sp_u) / np.sin(phi)
            vp   = np.cos(phi / 2) * sp_u + np.sin(phi / 2) * perp  # consistent
            vm   = np.cos(phi / 2) * sp_u - np.sin(phi / 2) * perp  # suppression
        # Primal weight recovery: w* = X' K_Z^+ v*
        w_plus  = X.T @ (U @ (inv1 * (U.T @ vp)))
        w_minus = X.T @ (U @ (inv1 * (U.T @ vm)))
        return (_canonical_sign(w_plus, X, A),
                _canonical_sign(w_minus, X, A),
                float(T_dual), df_dual, 'dual')

    # --- Primal: bisector reconstruction ---
    s_p = np.sqrt(max(float(a @ p_vec), 1e-15))
    c_pq = float(z @ p_vec) / s_p
    s_pq = np.sqrt(max(float(z @ q_vec) - c_pq**2, 0.0))
    cos_phi = c_pq / np.sqrt(c_pq**2 + s_pq**2) if (c_pq**2 + s_pq**2) > 1e-30 else 0.0
    cos_phi = float(np.clip(cos_phi, -1.0, 1.0))
    phi = float(np.arccos(cos_phi))
    df_primal = p - 1
    cos2 = cos_phi**2
    T_primal = cos_phi * np.sqrt(df_primal / max(1.0 - cos2, 1e-15))
    sp_u = p_vec / s_p
    sq_u = q_vec / np.sqrt(max(float(z @ q_vec), 1e-15))
    if abs(np.sin(phi)) < 1e-12:
        wp = sp_u; wm = -sq_u
    else:
        perp = (sq_u - cos_phi * sp_u) / np.sin(phi)
        wp   = np.cos(phi / 2) * sp_u + np.sin(phi / 2) * perp
        wm   = np.cos(phi / 2) * sp_u - np.sin(phi / 2) * perp
    return (_canonical_sign(wp, X, A),
            _canonical_sign(wm, X, A),
            float(T_primal), df_primal, 'primal')


def cosine_test(X, A, Y):
    """Global cosine test for the existence of any composite mediator.

    Tests H_0: max_w alpha(w)*beta(w) = 0, i.e. no linear combination of X
    mediates the effect of A on Y.

    Automatically selects the appropriate implementation based on dimensions:

    **Primal** (n > p): whitens w.r.t. V_Z = X'X - aa'/||A||^2 (p x p),
    solves V_Z p = a and V_Z q = z via Cholesky.

        cos(phi) = (a'q) / sqrt(a'p * z'q)
        T = cos(phi) * sqrt((p-1) / (1 - cos^2(phi)))  ~  t(p-1) under H0

    **Dual** (p >= n): operates in the n-dimensional sample space via the
    kernel K_Z = Z Z' (n x n) where Z = M_A X is the A-residualised design.
    Eigendecomposes K_Z = U diag(lambda) U', then whitens using lambda^{-1}:

        fa = U_r'(Z a),  fz = U_r'(Z z),  inv2 = 1/lambda^2
        cos(phi) = (fa . inv2*fz) / sqrt((fa . inv2*fa) * (fz . inv2*fz))
        T = cos(phi) * sqrt((n-2) / (1 - cos^2(phi)))  ~  t(n-2) under H0

    Parameters
    ----------
    X : (n, p) numeric array — need not be centred
    A : (n,)   numeric vector, treatment / exposure
    Y : (n,)   numeric vector, outcome

    Returns
    -------
    dict with keys:
        cos_phi   : float, cosine of angle between whitened path vectors
        T         : float, t-statistic
        p_value             : float, two-sided p-value (alias for p_value_two_sided)
        p_value_two_sided   : float, 2*P(t(df) >= |T|) -- detects any mediation
        p_value_consistent  : float, P(t(df) >= T)     -- one-sided, consistent mediation
        p_value_suppression : float, P(t(df) <= T)     -- one-sided, suppression
        df        : int, degrees of freedom (p-1 for primal, n-2 for dual)
        method    : str, 'primal' or 'dual'
    """
    from scipy.stats import t as t_dist

    X, A, Y = _centre(X, A, Y)
    n, p = X.shape
    a, z, AA = _sufficient_stats(X, A, Y)

    if n > p:
        # --- Primal: O(p^3) Cholesky on V_Z (p x p) ---
        VZ = X.T @ X - np.outer(a, a) / AA
        reg = max(1e-10 * np.trace(VZ) / p, 1e-10)
        fct = cho_factor(VZ + reg * np.eye(p), lower=True)
        p_vec = cho_solve(fct, a)
        q_vec = cho_solve(fct, z)

        s_p  = np.sqrt(max(float(a @ p_vec), 1e-15))
        c_pq = float(z @ p_vec) / s_p
        s_pq = np.sqrt(max(float(z @ q_vec) - c_pq**2, 0.0))
        denom   = np.sqrt(c_pq**2 + s_pq**2)
        cos_phi = float(c_pq / denom) if denom > 1e-15 else 0.0
        df = p - 1
        method = 'primal'

    else:
        # --- Dual: O(n^2 p) kernel on K_Z = Z Z' (n x n) ---
        # Dual path vectors: tilde_p = Z p = Z SZ^+ a = KZ^+(Z a)
        #                    tilde_q = Z q = Z SZ^+ z = KZ^+(Z z)
        # Cosine is the standard Euclidean angle between tilde_p and tilde_q,
        # which equals the SZ-metric angle (angle preservation identity).
        Z  = X - np.outer(A, (A @ X) / AA)
        KZ = Z @ Z.T
        eigvals, eigvecs = np.linalg.eigh(KZ)
        pos = eigvals > 1e-9 * max(float(eigvals[-1]), 1e-15)
        ev = eigvals[pos]; U = eigvecs[:, pos]
        # n-dim projections: Z a and Z z  (not K A and K Q_A Y)
        ta = Z @ (X.T @ A)                         # = Q_A K A
        tz = Z @ (X.T @ Y - (A @ Y / AA) * (X.T @ A))  # = Q_A K Q_A Y
        inv1 = 1.0 / ev
        tp = U @ (inv1 * (U.T @ ta))               # KZ^+ ta = tilde_p
        tq = U @ (inv1 * (U.T @ tz))               # KZ^+ tz = tilde_q
        # Euclidean cosine between tilde_p and tilde_q
        s_p = np.sqrt(max(float(tp @ tp), 1e-15))
        s_q = np.sqrt(max(float(tq @ tq), 1e-15))
        cos_phi = float(np.clip(float(tp @ tq) / (s_p * s_q), -1.0, 1.0))
        df = n - 2
        method = 'dual'

    cos2 = cos_phi**2
    if cos2 < 1.0 - 1e-15 and df > 0:
        T = cos_phi * np.sqrt(df / (1.0 - cos2))
    else:
        T = np.inf if cos_phi > 0 else -np.inf

    p_value_two_sided   = float(2.0 * t_dist.sf(abs(T), df=df))
    p_value_consistent  = float(t_dist.sf(T, df=df))        # one-sided, T > 0
    p_value_suppression = float(t_dist.cdf(T, df=df))       # one-sided, T < 0

    return {'cos_phi': cos_phi, 'T': T,
            'p_value':             p_value_two_sided,        # default = two-sided
            'p_value_two_sided':   p_value_two_sided,
            'p_value_consistent':  p_value_consistent,
            'p_value_suppression': p_value_suppression,
            'df': df, 'method': method}


def solve_maxcor(X, A, Y, Sigma_X=None):
    """MaxCor: semi-closed form solver that maximises the mediation index
    f*(w) = cor(X @ w, A) * cor(Z @ w, Y), where Z = X - A(A'A)^{-1}A'X.

    Whitens w.r.t. V_X = X'X, reduces the search to a 1-D angle, and
    refines with a grid search + bisection.
    Cost: O(np) sufficient statistics + O(p^3/3) Cholesky + O(1) 1-D search.

    Parameters
    ----------
    X       : (n, p) numeric array — need not be centred
    A       : (n,)   numeric vector, treatment / exposure
    Y       : (n,)   numeric vector, outcome
    Sigma_X : (p, p) known population covariance of X (optional, scale 1/n).

    Returns
    -------
    w : (p,) optimal weight vector, normalised so that cor(X @ w, A) >= 0.
    """
    X, A, Y = _centre(X, A, Y)
    a, z, AA = _sufficient_stats(X, A, Y)
    p = X.shape[1]
    if Sigma_X is not None:
        VX = np.asarray(Sigma_X, dtype=float)
        if VX.shape != (p, p):
            raise ValueError(f"Sigma_X shape {VX.shape} does not match p={p}")
        VX = VX * X.shape[0]
    else:
        VX  = X.T @ X
    reg = max(1e-10 * np.trace(VX) / p, 1e-10)
    fct = cho_factor(VX + reg * np.eye(p), lower=True)
    w = _reconstruct(a, z, AA, cho_solve(fct, a), cho_solve(fct, z))
    return _canonical_sign(w, X, A)


# =============================================================================
# 3. OLS baselines
# =============================================================================

def solve_reg_YX(X, A, Y, Sigma_X=None):
    """OLS baseline [3] Reg Y~X: sets w = V_X^{-1} X'Y.

    Maximises cor(X @ w, Y) without controlling for A.  Equivalent to
    w = V_X^{-1}(z + (A'Y/||A||^2) a), where z and a are the sufficient stats.
    This is the standard outcome-prediction direction (regress Y on X).

    Sigma_X : (p, p) known population covariance of X (optional, scale 1/n).
    """
    X, A, Y = _centre(X, A, Y)
    p  = X.shape[1]
    if Sigma_X is not None:
        VX = np.asarray(Sigma_X, dtype=float)
        if VX.shape != (p, p):
            raise ValueError(f"Sigma_X shape {VX.shape} does not match p={p}")
        VX = VX * X.shape[0]
    else:
        VX = X.T @ X
    reg = max(1e-10 * np.trace(VX) / p, 1e-10)
    fct = cho_factor(VX + reg * np.eye(p), lower=True)
    w   = cho_solve(fct, X.T @ Y)
    return _canonical_sign(w, X, A)


def solve_reg_AX(X, A, Y, Sigma_X=None):
    """OLS baseline [4] Reg A~X: sets w = V_X^{-1} a.

    Maximises cor(X @ w, A) — the treatment-mediator path — while ignoring
    the outcome Y.  Corresponds to a standard aging clock (regress age on
    proteins without reference to disease outcome).

    Sigma_X : (p, p) known population covariance of X (optional, scale 1/n).
    """
    X, A, Y = _centre(X, A, Y)
    a = X.T @ A
    p = X.shape[1]
    if Sigma_X is not None:
        VX = np.asarray(Sigma_X, dtype=float)
        if VX.shape != (p, p):
            raise ValueError(f"Sigma_X shape {VX.shape} does not match p={p}")
        VX = VX * X.shape[0]
    else:
        VX = X.T @ X
    reg = max(1e-10 * np.trace(VX) / p, 1e-10)
    fct = cho_factor(VX + reg * np.eye(p), lower=True)
    w   = cho_solve(fct, a)
    return _canonical_sign(w, X, A)


# =============================================================================
# 4. Numerical baselines (L-BFGS-B with random restarts, for validation only)
# =============================================================================

def solve_numerical_h(X, A, Y, n_restarts=10, seed=42):
    """L-BFGS-B maximiser of h = alpha_hat * beta_hat (numerical baseline)."""
    from scipy.optimize import minimize
    X, A, Y = _centre(X, A, Y)
    a, z, AA = _sufficient_stats(X, A, Y)
    VZ = X.T @ X - np.outer(a, a) / AA

    def neg_h(w):
        qZ = float(w @ (VZ @ w))
        if qZ < 1e-15: return 0.0
        return -((float(w @ a) / AA) * (float(w @ z) / qZ))

    p   = X.shape[1]
    rng = np.random.default_rng(seed)
    best_val, best_w = np.inf, np.ones(p) / np.sqrt(p)
    for _ in range(n_restarts):
        w0  = rng.standard_normal(p)
        nrm = np.linalg.norm(w0)
        w0  = w0 / nrm if nrm > 1e-15 else np.ones(p) / np.sqrt(p)
        res = minimize(neg_h, w0, method='L-BFGS-B')
        if res.fun < best_val:
            best_val, best_w = res.fun, res.x
    return _canonical_sign(best_w, X, A)


def solve_numerical_fstar(X, A, Y, n_restarts=10, seed=42):
    """L-BFGS-B maximiser of f* = cor(Xw,A)*cor(Zw,Y) (numerical baseline)."""
    from scipy.optimize import minimize
    X, A, Y = _centre(X, A, Y)
    Z = X - np.outer(A, (A @ X) / (A @ A))

    def cor(u, v):
        u, v = u - u.mean(), v - v.mean()
        return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15)

    def neg_fstar(w):
        return -(cor(X @ w, A) * cor(Z @ w, Y))

    p   = X.shape[1]
    rng = np.random.default_rng(seed)
    best_val, best_w = np.inf, np.ones(p) / np.sqrt(p)
    for _ in range(n_restarts):
        w0  = rng.standard_normal(p)
        nrm = np.linalg.norm(w0)
        w0  = w0 / nrm if nrm > 1e-15 else np.ones(p) / np.sqrt(p)
        res = minimize(neg_fstar, w0, method='L-BFGS-B')
        if res.fun < best_val:
            best_val, best_w = res.fun, res.x
    return _canonical_sign(best_w, X, A)


# =============================================================================
# 5. DM baseline  (Chén et al. 2018 — Directions of Mediation)
# =============================================================================
#
# Python re-implementation of hdmed::mediate_hdmm (CRAN / oliverychen/PDM).
#
# The first Direction of Mediation (DM1) is the unit-norm vector w that
# maximises the joint log-likelihood of the two-equation LSEM:
#
#   m_i = a0 + a1*A_i + eps_i,    eps ~ N(0, s2a)
#   Y_i = b0 + g*A_i + b*m_i + xi_i,  xi ~ N(0, s2b)
#
# where m = Xw is the latent mediator.  Concentrating out scalar parameters
# by OLS at each step, the w-update reduces to a sphere-constrained QP:
#
#   max  w'c - (1/2) w'Qw     s.t.  ||w|| = 1
#
# solved via eigendecomposition of Q = X'X and lambda bisection.
#
# Reference: Chén OY et al. (2018). High-dimensional multivariate mediation
#   with application to neuroimaging data. Biostatistics 19(2):121–136.

def solve_dm(X, A, Y, tol=1e-5, imax=100, interval=1e6, step=1e4):
    """First Direction of Mediation (Chén et al. 2018).

    Python re-implementation of hdmed::mediate_hdmm (CRAN / oliverychen/PDM).
    Finds the unit-norm w that maximises the joint LSEM log-likelihood via
    block coordinate ascent: alternating OLS theta-updates and sphere-
    constrained QP w-updates solved by eigendecomposition + lambda bisection.

    Parameters
    ----------
    X        : (n, p) mediator matrix
    A        : (n,)   exposure vector
    Y        : (n,)   outcome vector
    tol      : convergence tolerance on ||w_new - w_old||  (default 1e-5)
    imax     : maximum outer iterations                     (default 100)
    interval : search radius for the lambda grid            (default 1e6)
    step     : coarse grid step for initial lambda search   (default 1e4)

    Returns
    -------
    w : (p,) unit-norm weight vector, sign-corrected so alpha_hat(w) >= 0.
    """
    Xc, Ac, Yc = _centre(X, A, Y)
    n, p = Xc.shape

    ones = np.ones(n)
    DA   = np.column_stack([ones, Ac])
    MtM  = Xc.T @ Xc
    eigvals_MtM, eigvecs_MtM = np.linalg.eigh(MtM)

    w = np.ones(p) / np.sqrt(p)

    for _ in range(imax):
        m = Xc @ w

        c_a, _, _, _ = np.linalg.lstsq(DA, m, rcond=None)
        hat_m = DA @ c_a
        r_a   = m - hat_m
        s2a   = max(float(r_a @ r_a) / n, 1e-10)

        DYm = np.column_stack([ones, Ac, m])
        c_y, _, _, _ = np.linalg.lstsq(DYm, Yc, rcond=None)
        b         = float(c_y[2])
        hat_y_nom = DA @ c_y[:2]
        Y_res     = Yc - hat_y_nom
        r_b       = Yc - DYm @ c_y
        s2b       = max(float(r_b @ r_b) / n, 1e-10)

        lam_Q  = 1.0/s2a + b**2/s2b
        eig_Q  = lam_Q * eigvals_MtM
        c_vec  = (1.0/s2a) * (Xc.T @ hat_m) + (b/s2b) * (Xc.T @ Y_res)
        c_rot  = eigvecs_MtM.T @ c_vec

        def f_norm_sq(lam):
            d = eig_Q + lam
            d = np.where(np.abs(d) < 1e-12, 1e-12, d)
            return float(np.sum((c_rot / d)**2))

        lam0 = max(-eig_Q[0], 0.0) + 1e-6
        if f_norm_sq(lam0) <= 1.0:
            # Unconstrained solution already inside unit sphere; constraint inactive.
            lam_star = lam0
        else:
            lo, hi = lam0, lam0 + interval
            if f_norm_sq(hi) > 1.0:
                hi = lam0 + interval * 100  # extend if needed
            for _ in range(80):
                mid = (lo + hi) / 2.0
                if f_norm_sq(mid) > 1.0: lo = mid
                else:                    hi = mid
            lam_star = (lo + hi) / 2.0

        d     = eig_Q + lam_star
        d     = np.where(np.abs(d) < 1e-12, 1e-12, d)
        w_new = eigvecs_MtM @ (c_rot / d)
        nrm   = np.linalg.norm(w_new)
        if nrm > 1e-12:
            w_new = w_new / nrm

        delta = np.linalg.norm(w_new - w)
        w     = w_new
        if delta < tol:
            break

    return _canonical_sign(w, Xc, Ac)


# =============================================================================
# 6. Mediation statistics
# =============================================================================

def mediation_stats(M, A, Y):
    """Compute conventional mediation statistics for a composite mediator M = X @ w.

    Follows the Baron & Kenny / product-of-coefficients framework using OLS
    regression coefficients, so the results match standard mediation packages
    (e.g. R's mediation package) exactly.

    Three OLS regressions are fitted (all on centred variables):
        (1)  M ~ A          -> alpha  (treatment -> mediator path)
        (2)  Y ~ A + M      -> gamma  (direct effect), beta (mediator -> outcome)
        (3)  Y ~ A          -> tau    (total effect)

    M is standardised to unit variance internally before computing path
    coefficients, making alpha, beta, and gamma scale-free and directly
    comparable across methods with different composite scales.
    Correlation-based quantities (f*, cor_MA, cor_MY_res) are unaffected.

    Parameters
    ----------
    M : (n,) composite mediator score, e.g. X_test @ w
    A : (n,) treatment / exposure vector
    Y : (n,) outcome vector

    Returns
    -------
    pandas.DataFrame with columns Symbol, Description, Value and rows:
        mediation_index, cor_MA, cor_MY_res, alpha, beta, gamma,
        total_effect, indirect_effect, prop_mediated, sobel_z, sobel_p
    """
    import pandas as pd
    from scipy import stats as _stats
    import sys

    M = np.asarray(M, dtype=float).ravel()
    A = np.asarray(A, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    if not (len(M) == len(A) == len(Y)):
        raise ValueError(
            f"mediation_stats: M, A, Y must have the same length "            f"(got {len(M)}, {len(A)}, {len(Y)}).")
    if len(A) < 3:
        raise ValueError("mediation_stats requires at least 3 observations.")

    M = M - M.mean(); A = A - A.mean(); Y = Y - Y.mean()
    M_sd = np.std(M, ddof=1)
    if M_sd > 1e-12:
        M = M / M_sd

    def _cor(u, v):
        return float((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15))

    def _ols1(y, x):
        return float((x @ y) / (x @ x + 1e-15))

    def _ols2(y, x1, x2):
        X2 = np.column_stack([x1, x2])
        return np.linalg.solve(X2.T @ X2 + 1e-10 * np.eye(2), X2.T @ y)

    M_res      = M - A * _ols1(M, A)
    cor_MA     = _cor(M, A)
    cor_MY_res = _cor(M_res, Y)
    objective  = cor_MA * cor_MY_res

    n               = len(A)
    alpha           = _ols1(M, A)
    gamma, beta     = _ols2(Y, A, M)
    total_effect    = _ols1(Y, A)
    indirect_effect = alpha * beta
    direct_effect   = gamma
    prop_mediated   = (indirect_effect / total_effect
                       if abs(total_effect) > 1e-12 else float('nan'))

    resid_alpha  = M - alpha * A
    sigma2_alpha = float(resid_alpha @ resid_alpha) / (n - 1)
    se_alpha     = float(np.sqrt(sigma2_alpha / (A @ A + 1e-15)))

    Xmat       = np.column_stack([A, M])
    coefs      = np.array([gamma, beta])
    resid_beta = Y - Xmat @ coefs
    sigma2_beta = float(resid_beta @ resid_beta) / (n - 2)
    cov_coefs   = sigma2_beta * np.linalg.solve(Xmat.T @ Xmat + 1e-10 * np.eye(2), np.eye(2))
    se_beta     = float(np.sqrt(cov_coefs[1, 1]))

    sobel_se = float(np.sqrt(beta**2 * se_alpha**2 + alpha**2 * se_beta**2))
    sobel_z  = float(indirect_effect / sobel_se) if sobel_se > 1e-15 else float('nan')
    sobel_p  = float(2 * (1 - _stats.norm.cdf(abs(sobel_z)))) if not np.isnan(sobel_z) else float('nan')

    def _fmt_p(p):
        if np.isnan(p): return float('nan')
        if p == 0.0:    return f"<{sys.float_info.min:.1e}"
        return f"{p:.3e}"

    values_rounded = [
        _fmt_p(sobel_p) if k == "sobel_p" else round(float(v), 4)
        for k, v in zip(
            ["objective","cor_MA","cor_MY_res","alpha","beta","gamma",
             "total_effect","indirect_effect","prop_mediated","sobel_z","sobel_p"],
            [objective, cor_MA, cor_MY_res, alpha, beta, direct_effect,
             total_effect, indirect_effect, prop_mediated, sobel_z, sobel_p]
        )
    ]

    return pd.DataFrame(
        {
            "Symbol": [
                "f*", "r_MA", "r_M\u22a5Y",
                "\u03b1", "\u03b2", "\u03b3",
                "\u03c4", "\u03b1\u03b2", "\u03b1\u03b2/\u03c4",
                "Z", "p",
            ],
            "Description": [
                "Mediation index: cor(M,A)\u00b7cor(M_res,Y)",
                "cor(M, A)  \u2014 treatment\u2013mediator",
                "cor(M_res, Y)  \u2014 mediator\u2013outcome (net of A)",
                "Treatment \u2192 mediator  (M ~ A)",
                "Mediator \u2192 outcome  (Y ~ A + M, coeff on M)",
                "Direct effect  (Y ~ A + M, coeff on A)",
                "Total effect  (Y ~ A)",
                "Indirect effect  (\u03b1 \u00d7 \u03b2)",
                "Proportion mediated  (\u03b1\u03b2 / \u03c4)",
                "Sobel test statistic",
                "Sobel two-sided p-value",
            ],
            "Value": values_rounded,
        },
        index=[
            "mediation_index", "cor_MA", "cor_MY_res",
            "alpha", "beta", "gamma",
            "total_effect", "indirect_effect", "prop_mediated",
            "sobel_z", "sobel_p",
        ],
    )


# =============================================================================
# 7. Data generation
# =============================================================================

def generate_data(n, p, seed, scenario, sigma=0.5, rho=0.75, tau=0.25):
    """Generate one dataset under the specified scenario.

    Parameters
    ----------
    n        : number of observations
    p        : number of features
    seed     : integer random seed
    scenario : 'independent' | 'quarter' | 'partial' | 'same'
    sigma    : noise std dev for A and Y (default 0.5)
    rho      : AR(1) correlation parameter for X (default 0.75)
    tau      : direct effect of A on Y (default 0.25)

    Returns
    -------
    X, Z, A, Y  : centred arrays; Z = M_A X (residual projection of X on A)
    alpha, beta : true signal vectors (unit norm, p/4 nonzeros each)

    Scenarios (S1--S4 in the paper)
    --------------------------------
    S1 'independent' : alpha and beta on independent random p/4 supports
                       (expected cosine similarity ~0)
    S2 'quarter'     : p/4 nonzeros each; p/16 shared indices with identical
                       pre-normalisation values; rest unique to each vector
                       (expected cosine similarity ~0.23)
    S3 'partial'     : p/4 nonzeros each; p/8 shared indices with identical
                       pre-normalisation values; rest unique to each vector
                       (expected cosine similarity ~0.47)
    S4 'same'        : beta = alpha (identical signal directions,
                       cosine similarity = 1)
    """
    rng = np.random.default_rng(seed)
    p4  = max(1, p // 4)    # nonzeros per vector
    p8  = max(1, p // 8)    # shared nonzeros for S3
    p16 = max(1, p // 16)   # shared nonzeros for S2

    if scenario == 'independent':
        idx_a = rng.choice(p, size=p4, replace=False)
        idx_b = rng.choice(p, size=p4, replace=False)
        alpha = np.zeros(p); alpha[idx_a] = rng.standard_normal(p4)
        alpha /= np.linalg.norm(alpha)
        beta  = np.zeros(p); beta[idx_b]  = rng.standard_normal(p4)
        beta  /= np.linalg.norm(beta)

    elif scenario == 'quarter':
        idx        = rng.permutation(p)
        shared     = idx[:p16]
        alpha_only = idx[p16:p4]
        beta_only  = idx[p4: p4 + (p4 - p16)]
        sv         = rng.standard_normal(p16)
        alpha = np.zeros(p); alpha[shared] = sv
        alpha[alpha_only] = rng.standard_normal(p4 - p16)
        alpha /= np.linalg.norm(alpha)
        beta  = np.zeros(p); beta[shared] = sv
        beta[beta_only] = rng.standard_normal(p4 - p16)
        beta  /= np.linalg.norm(beta)

    elif scenario == 'partial':
        idx        = rng.permutation(p)
        shared     = idx[:p8]
        alpha_only = idx[p8:p4]
        beta_only  = idx[p4: p4 + (p4 - p8)]
        sv         = rng.standard_normal(p8)
        alpha = np.zeros(p); alpha[shared] = sv
        alpha[alpha_only] = rng.standard_normal(p4 - p8)
        alpha /= np.linalg.norm(alpha)
        beta  = np.zeros(p); beta[shared] = sv
        beta[beta_only] = rng.standard_normal(p4 - p8)
        beta  /= np.linalg.norm(beta)

    elif scenario == 'same':
        idx_a = rng.choice(p, size=p4, replace=False)
        alpha = np.zeros(p); alpha[idx_a] = rng.standard_normal(p4)
        alpha /= np.linalg.norm(alpha)
        beta  = alpha.copy()

    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    idx_p = np.arange(p)
    Sigma = rho ** np.abs(idx_p[:, None] - idx_p[None, :])
    L     = np.linalg.cholesky(Sigma)
    X     = rng.standard_normal((n, p)) @ L.T; X -= X.mean(0)

    A = X @ alpha + sigma * rng.standard_normal(n); A -= A.mean()
    Y = X @ beta + tau * A + sigma * rng.standard_normal(n); Y -= Y.mean()
    Z = X - np.outer(A, (A @ X) / (A @ A))

    return X, Z, A, Y, alpha, beta


# =============================================================================
# 8. Worked example
# =============================================================================

def worked_example():
    """Demonstrate the solvers on an S3 dataset (50% overlap, n=200, p=20, seed=0)."""
    X, Z, A, Y, alpha, beta = generate_data(n=200, p=20, seed=0, scenario='partial')
    print(f"cos(alpha, beta) = {alpha @ beta:.3f}")

    w_maxie, _, _, _, _ = solve_maxie(X, A, Y)  # consistent weight
    w_maxcor = solve_maxcor(X, A, Y)
    w_ryx    = solve_reg_YX(X, A, Y)
    w_rax    = solve_reg_AX(X, A, Y)

    def cor(u, v):
        u, v = u - u.mean(), v - v.mean()
        return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))

    def fstar(w):
        return cor(X @ w, A) * cor(Z @ w, Y)

    def h(w):
        a  = X.T @ A;  AA = float(A @ A)
        z  = X.T @ Y - (float(A @ Y) / AA) * a
        VZ = X.T @ X - np.outer(a, a) / AA
        qZ = float(w @ (VZ @ w))
        return 0.0 if qZ < 1e-15 else (float(w @ a) / AA) * (float(w @ z) / qZ)

    print(f"\n  Method        h=alpha*beta     f*")
    print(f"  ----------  ------------  -------")
    for name, w in [("[1] MaxIE",   w_maxie),  ("[2] MaxCor", w_maxcor),
                    ("[3] Reg Y~X", w_ryx),    ("[4] Reg A~X", w_rax)]:
        print(f"  {name:<12}  {h(w):12.6f}  {fstar(w):7.6f}")


# =============================================================================
# 9. Paper outputs — one function per table / figure
# =============================================================================

# Shared simulation constants and helpers used across all table/figure functions
_SCENARIOS   = [('independent','S1'), ('quarter','S2'),
                ('partial','S3'),     ('same','S4')]
_OBJ_CONFIGS = [(500, 20), (1000, 100), (1000, 500), (500, 1000)]  # last row is dual (p>=n)
_OBJ_METHODS = [
    ('[1] MaxIE',   solve_maxie),
    ('[2] MaxCor',  solve_maxcor),
    ('[3] Reg Y~X', solve_reg_YX),
    ('[4] Reg A~X', solve_reg_AX),
    ('[5] DM',      solve_dm),
]

def _cor(u, v):
    u, v = u - u.mean(), v - v.mean()
    return float((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15))

def _obj_h(w, X, A, Y):
    a  = X.T @ A;  AA = float(A @ A)
    z  = X.T @ Y - (float(A @ Y) / AA) * a
    VZ = X.T @ X - np.outer(a, a) / AA
    qZ = float(w @ (VZ @ w))
    return 0.0 if qZ < 1e-15 else (float(w @ a) / AA) * (float(w @ z) / qZ)

def _obj_fstar(w, X, Z, A, Y):
    return _cor(X @ w, A) * _cor(Z @ w, Y)

def _collect_objectives(n_seeds=20):
    """Run all methods on all (scenario, n, p) cells; return nested dict.
    For p >= n rows, only MaxIE (dual) is run; MaxCor/OLS/DM return None.
    """
    cache = {}
    for sc_key, sc_label in _SCENARIOS:
        for n, p in _OBJ_CONFIGS:
            dual_row = (p >= n)
            h_res = {k: [] for k, _ in _OBJ_METHODS}
            f_res = {k: [] for k, _ in _OBJ_METHODS}
            for seed in range(n_seeds):
                X, Z, A, Y, _, _ = generate_data(n, p, seed, sc_key)
                for label, fn in _OBJ_METHODS:
                    if dual_row and label != '[1] MaxIE':
                        h_res[label].append(None)
                        f_res[label].append(None)
                        continue
                    result = fn(X, A, Y)
                    w = result[0] if isinstance(result, tuple) else result
                    h_res[label].append(_obj_h(w, X, A, Y))
                    f_res[label].append(_obj_fstar(w, X, Z, A, Y))
            cache[(sc_key, n, p)] = {'h': h_res, 'f': f_res, 'dual': dual_row}
    return cache


# =============================================================================
# 9a. Table tab:obj_h — mean indirect effect h across scenarios
# =============================================================================

def table_obj_h(n_seeds=20):
    """Reproduce Table tab:obj_h: mean h = alpha_hat*beta_hat (std).
    S1--S4 x four (n,p) configs x five methods; dual row (p>=n) shows MaxIE only.
    """
    cache = _collect_objectives(n_seeds)
    cw  = 14
    fmt = lambda v: f'{np.mean(v):.2f}({np.std(v):.2f})'
    na_span = ' ' * (cw * 4) + f'{"n/a (dual, p>=n)":>{cw}}'
    hdr = f"{'Setting':<24}" + "".join(f"{k:>{cw}}" for k, _ in _OBJ_METHODS)
    print("\nTable tab:obj_h: Mean h = alpha_hat*beta_hat (std)")
    print(hdr); print("-" * len(hdr))
    for sc_key, sc_label in _SCENARIOS:
        for n, p in _OBJ_CONFIGS:
            entry = cache[(sc_key, n, p)]
            h_res = entry['h']
            dual  = entry['dual']
            row = f"{sc_label} n={n:4d} p={p:4d}  "
            if dual:
                maxie_str = f"{fmt(h_res['[1] MaxIE']):>{cw}}"
                row += maxie_str + f"{'n/a (dual, p>=n)':>{cw*4}}"
            else:
                row += "".join(f"{fmt(h_res[k]):>{cw}}" for k, _ in _OBJ_METHODS)
            print(row)
        print()
    return cache


# =============================================================================
# 9b. Table tab:obj_f — mean mediation index f* across scenarios
# =============================================================================

def table_obj_f(cache=None, n_seeds=20):
    """Reproduce Table tab:obj_f: mean f* = cor(Xw,A)*cor(Zw,Y) (std).
    Pass the cache returned by table_obj_h to avoid rerunning simulations.
    """
    if cache is None:
        cache = _collect_objectives(n_seeds)
    cw  = 14
    fmt = lambda v: f'{np.mean(v):.2f}({np.std(v):.2f})'
    hdr = f"{'Setting':<24}" + "".join(f"{k:>{cw}}" for k, _ in _OBJ_METHODS)
    print("\nTable tab:obj_f: Mean f* (std)")
    print(hdr); print("-" * len(hdr))
    for sc_key, sc_label in _SCENARIOS:
        for n, p in _OBJ_CONFIGS:
            entry = cache[(sc_key, n, p)]
            f_res = entry['f']
            dual  = entry['dual']
            row = f"{sc_label} n={n:4d} p={p:4d}  "
            if dual:
                maxie_str = f"{fmt(f_res['[1] MaxIE']):>{cw}}"
                row += maxie_str + f"{'n/a (dual, p>=n)':>{cw*4}}"
            else:
                row += "".join(f"{fmt(f_res[k]):>{cw}}" for k, _ in _OBJ_METHODS)
            print(row)
        print()


# =============================================================================
# 9c. Table tab:time — wall-clock timing vs numerical baselines (S3, n=500)
# =============================================================================

def _system_info():
    """Print CPU, RAM, Python/NumPy/SciPy versions for reporting in Table tab:time."""
    import platform, sys
    import numpy as np
    from scipy import __version__ as sp_ver

    import subprocess

    # CPU: macOS via sysctl, Linux via /proc/cpuinfo, fallback via platform
    cpu = "unknown"
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True).strip()
        except Exception:
            pass
    if cpu == "unknown":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line and "unknown" not in line.lower():
                        cpu = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    if cpu == "unknown":
        cpu = platform.processor() or platform.machine() or "unknown"

    # RAM: macOS via sysctl, Linux via /proc/meminfo, fallback via psutil
    ram = "unknown"
    if platform.system() == "Darwin":
        try:
            bytes_ = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True).strip())
            ram = f"{bytes_ / 1024**3:.0f} GB"
        except Exception:
            pass
    if ram == "unknown":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        ram = f"{kb / 1024**2:.1f} GB"
                        break
        except OSError:
            try:
                import psutil
                ram = f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
            except ImportError:
                pass

    print("Computing environment")
    print(f"  CPU    : {cpu}")
    print(f"  RAM    : {ram}")
    print(f"  OS     : {platform.system()} {platform.release()}")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  NumPy  : {np.__version__}")
    print(f"  SciPy  : {sp_ver}")
    print()


def table_time(configs=_OBJ_CONFIGS, n_seeds=5, n_trials=5):
    """Reproduce Table tab:time: closed-form vs numerical baselines vs DM timing.

    Uses S3 (50% overlap) for each (n, p) config in configs.
    Numerical solvers run only for p < 500; DM skipped for p >= n (dual row).

    Parameters
    ----------
    configs  : list of (n, p) tuples; default matches Tables 2/3 scenarios
    n_seeds  : replicates per config (default 5)
    n_trials : batch size for closed-form timing (default 5)
    """
    import time as _time

    _system_info()

    ow = 14   # column width for objective panel
    tw = 11   # column width for timing panel
    num_configs  = [(n, p) for n, p in configs if p < 500]
    primal_configs = [(n, p) for n, p in configs if p < n]   # primal only rows

    obj_cols = ['[1] MaxIE', 'Num-h', '[2] MaxCor', 'Num-f*']
    tim_cols = ['[1] MaxIE', 'Num-h', '[2] MaxCor', 'Num-f*',
                '[3] RegY~X', '[4] RegA~X', '[5] DM']
    obj_hdr_body = ''.join(f"{c:>{ow}}" for c in obj_cols)
    tim_hdr_body = ''.join(f"{c:>{tw}}" for c in tim_cols)

    row_labels = [f"n={n:4d} p={p:4d}" for n, p in configs]
    lw = max(len(s) for s in row_labels) + 1

    # warm-up (skip DM and numerical for dual rows)
    for n, p in configs:
        Xw, _, Aw, Yw, _, _ = generate_data(n, p, 99, 'partial')
        for fn in [solve_maxie, solve_maxcor, solve_reg_YX, solve_reg_AX]:
            r = fn(Xw, Aw, Yw); _ = r[0] if isinstance(r, tuple) else r
        if p < n:
            solve_dm(Xw, Aw, Yw)

    def tms_batch(fn, X, A, Y, k=n_trials):
        w = None
        t0 = _time.perf_counter()
        for _ in range(k):
            r = fn(X, A, Y); w = r[0] if isinstance(r, tuple) else r
        return (_time.perf_counter() - t0) * 1e3 / k, w

    obj_rows, tim_rows = [], []

    for n, p in configs:
        run_num  = (n, p) in num_configs
        dual_row = (p >= n)
        h_ie = []; h_nh = []; f_mc = []; f_nf = []
        t_ie = []; t_nh = []; t_mc = []; t_nf = []
        t_ryx = []; t_rax = []; t_dm = []

        for seed in range(n_seeds):
            X, Z, A, Y, _, _ = generate_data(n, p, seed, 'partial')

            ms, res = tms_batch(solve_maxie, X, A, Y); w = res[0]; t_ie.append(ms)
            h_ie.append(_obj_h(w, X, A, Y))

            if not dual_row:
                ms, w = tms_batch(solve_maxcor, X, A, Y); t_mc.append(ms)
                f_mc.append(_obj_fstar(w, X, Z, A, Y))
                ms, _ = tms_batch(solve_reg_YX, X, A, Y); t_ryx.append(ms)
                ms, _ = tms_batch(solve_reg_AX, X, A, Y); t_rax.append(ms)
                ms, _ = tms_batch(solve_dm,     X, A, Y); t_dm.append(ms)

            if run_num:
                ms, w = tms_batch(solve_numerical_h,     X, A, Y, k=1)
                t_nh.append(ms); h_nh.append(_obj_h(w, X, A, Y))
                if not dual_row:
                    ms, w = tms_batch(solve_numerical_fstar, X, A, Y, k=1)
                    t_nf.append(ms); f_nf.append(_obj_fstar(w, X, Z, A, Y))

        obj_rows.append((run_num, dual_row, h_ie, h_nh, f_mc, f_nf))
        tim_rows.append((
            np.mean(t_ie),
            np.mean(t_nh)  if t_nh  else None,
            np.mean(t_mc)  if t_mc  else None,
            np.mean(t_nf)  if t_nf  else None,
            np.mean(t_ryx) if t_ryx else None,
            np.mean(t_rax) if t_rax else None,
            np.mean(t_dm)  if t_dm  else None,
        ))

    fmt2 = lambda v: f"{np.mean(v):.2f}({np.std(v):.2f})"
    NA = 'n/a'

    # ---- Panel 1: Objective values ----
    print("\nTable tab:time: S3 (50% overlap)")
    print(f"\nObjective value (mean (sd) over {n_seeds} replicates; p=500 numerical omitted)")
    print(f"{'':>{lw}}{obj_hdr_body}")
    print("-" * (lw + len(obj_hdr_body)))
    for i, (n, p) in enumerate(configs):
        run_num, dual_row, h_ie, h_nh, f_mc, f_nf = obj_rows[i]
        if dual_row:
            # only MaxIE has data; rest are n/a
            print(f"{row_labels[i]:<{lw}}"
                  + f"{fmt2(h_ie):>{ow}}"
                  + f"{'n/a':>{ow}}"
                  + f"{'n/a (dual, p>=n)':>{ow*2}}")
        elif run_num:
            print(f"{row_labels[i]:<{lw}}"
                  + ''.join(f"{s:>{ow}}" for s in
                             [fmt2(h_ie), fmt2(h_nh), fmt2(f_mc), fmt2(f_nf)]))

    # ---- Panel 2: Wall-clock time ----
    print("\nWall-clock time (ms)")
    print(f"{'':>{lw}}{tim_hdr_body}")
    print("-" * (lw + len(tim_hdr_body)))
    for i, (n, p) in enumerate(configs):
        vals = tim_rows[i]
        cells = []
        for v in vals:
            if v is None:
                cells.append(f"{'n/a':>{tw}}" if p >= n else f"{'---':>{tw}}")
            else:
                cells.append(f"{v:{tw}.2f}")
        print(f"{row_labels[i]:<{lw}}" + ''.join(cells))
# =============================================================================
# 9c. Figure fig:power — three-panel empirical validation of Proposition 2
# =============================================================================

def figure_power_props(n_sims=1000, alpha=0.05, seed=0,
                       save_path='power_figure.pdf'):
    """Reproduce Figure 2 (fig:power): three-panel power validation.

    Left   : convergence mean(T) -> delta(phi0, p) as n grows  (p=40 fixed)
    Centre : power vs phi0 at p=40, n in {100, 200, 1000}
    Right  : power vs n   at p=40, three signal angles (55°, 84°, 90°)

    DGP: X ~ N(0,I_p), A = X[:,0]*0.5 + noise, Y = X[:,0]*0.5*cos(phi) +
         X[:,1]*0.5*sin(phi) + noise, tau=0.  Population phi0 computed
         analytically from the population SZ at the true path vectors.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure_power_props.")
        return

    from scipy.stats import t as t_dist
    from scipy.linalg import cho_factor, cho_solve

    rng   = np.random.default_rng(seed)
    sigma = 0.5
    scale = 0.5

    C3_CONV  = ['#377EB8', '#E41A1C', '#4DAF4A']
    C3_N     = ['#1B7837', '#762A83', '#E08214']
    C3_RIGHT = ['#377EB8', '#E41A1C', '#4DAF4A']
    LS3      = ['-', '--', ':']

    def _make_paths(p, phi_l2):
        a_s = np.zeros(p); a_s[0] = scale
        b_s = np.zeros(p); b_s[0] = scale * np.cos(phi_l2)
        if p > 1: b_s[1] = scale * np.sin(phi_l2)
        return a_s, b_s

    def _delta_theory(a_s, b_s, p):
        EA2    = float(a_s @ a_s) + sigma**2
        c_proj = float(a_s @ b_s) / EA2
        z0     = b_s - c_proj * a_s
        u      = a_s / np.sqrt(EA2)
        uu     = float(u @ u)
        def SZinv(v): return v + u * (float(u @ v) / max(1.0 - uu, 1e-15))
        pv = SZinv(a_s); qv = SZinv(z0)
        s_p  = np.sqrt(max(float(a_s @ pv), 1e-15))
        c_pq = float(z0 @ pv) / s_p
        s_pq2 = max(float(z0 @ qv) - c_pq**2, 0.0)
        denom = np.sqrt(c_pq**2 + s_pq2)
        cos_p = float(c_pq / denom) if denom > 1e-15 else 0.0
        phi_p = float(np.arccos(np.clip(cos_p, -1, 1)))
        delta = cos_p / max(np.sqrt(1.0 - cos_p**2), 1e-12) * np.sqrt(p - 1)
        return delta, phi_p

    def _sim_T(n, p, a_s, b_s):
        X = rng.standard_normal((n, p))
        A = X @ a_s + sigma * rng.standard_normal(n)
        Y = X @ b_s + sigma * rng.standard_normal(n)
        X = X - X.mean(0); A = A - A.mean(); Y = Y - Y.mean()
        AA = float(A @ A)
        a  = X.T @ A
        z  = X.T @ Y - (A @ Y / AA) * a
        VZ  = X.T @ X - np.outer(a, a) / AA
        reg = max(1e-10 * np.trace(VZ) / p, 1e-10)
        fct = cho_factor(VZ + reg * np.eye(p), lower=True)
        pv  = cho_solve(fct, a); qv = cho_solve(fct, z)
        s_p  = np.sqrt(max(float(a @ pv), 1e-15))
        c_pq = float(z @ pv) / s_p
        s_pq = np.sqrt(max(float(z @ qv) - c_pq**2, 0.0))
        denom = np.sqrt(c_pq**2 + s_pq**2)
        cos_phi = float(c_pq / denom) if denom > 1e-15 else 0.0
        cos2 = cos_phi**2
        T = cos_phi * np.sqrt((p-1) / (1.0 - cos2)) if cos2 < 1.0 - 1e-15 else (
            np.inf if cos_phi > 0 else -np.inf)
        return float(T)

    def _power_at(n, p, a_s, b_s):
        t_crit = t_dist.ppf(1.0 - alpha / 2, df=p - 1)
        return float(np.mean([abs(_sim_T(n, p, a_s, b_s)) >= t_crit
                               for _ in range(n_sims)]))

    # parameters
    P_A = 40
    N_CONV = [50, 100, 200, 400, 800, 1600, 3200]
    PHI_CONV_L2 = [np.radians(55), np.radians(70), np.radians(84)]

    P_B = 40
    N_POWER_B = [100, 200, 1000]
    eps = 0.08
    PHI_GRID = np.linspace(eps, np.pi - eps, 25)

    P_C = 40
    N_RIGHT = [50, 100, 200, 400, 800, 1600]
    PHI_ABOVE_L2 = np.radians(55)
    PHI_BELOW_L2 = np.radians(84)

    # Panel A
    print("  Panel A: convergence ...")
    mean_T_conv = np.zeros((len(PHI_CONV_L2), len(N_CONV)))
    std_T_conv  = np.zeros((len(PHI_CONV_L2), len(N_CONV)))
    delta_conv, phi_labels_conv = [], []
    for i, phi_l2 in enumerate(PHI_CONV_L2):
        a_s, b_s = _make_paths(P_A, phi_l2)
        delta, phi_p = _delta_theory(a_s, b_s, P_A)
        delta_conv.append(delta)
        phi_labels_conv.append(
            rf'$\varphi_0={phi_p/np.pi:.2f}\pi$  ($\delta={delta:.1f}$)')
        for j, n in enumerate(N_CONV):
            Ts = np.array([_sim_T(n, P_A, a_s, b_s) for _ in range(n_sims)])
            mean_T_conv[i, j] = float(np.mean(Ts))
            std_T_conv[i, j]  = float(np.std(Ts))

    # Panel B
    print("  Panel B: power vs phi0 ...")
    t_crit_b = t_dist.ppf(1.0 - alpha / 2, df=P_B - 1)
    power_phi = np.zeros((len(N_POWER_B), len(PHI_GRID)))
    for i, n in enumerate(N_POWER_B):
        for fi, phi_l2 in enumerate(PHI_GRID):
            a_s, b_s = _make_paths(P_B, phi_l2)
            power_phi[i, fi] = _power_at(n, P_B, a_s, b_s)
        print(f"    n={n} done")
    phi_half = np.linspace(0.01, np.pi/2, 500)
    delta_half = [_delta_theory(*_make_paths(P_B, ph), P_B)[0] for ph in phi_half]
    above_mask = np.array(delta_half) >= t_crit_b
    phi_thr = float(phi_half[above_mask].max()) if above_mask.any() else np.nan

    # Panel C
    print("  Panel C: power vs n ...")
    t_crit_c = t_dist.ppf(1.0 - alpha / 2, df=P_C - 1)
    a_ab, b_ab = _make_paths(P_C, PHI_ABOVE_L2)
    a_bl, b_bl = _make_paths(P_C, PHI_BELOW_L2)
    a_nu, b_nu = np.zeros(P_C), np.zeros(P_C)
    d_above, _ = _delta_theory(a_ab, b_ab, P_C)
    d_below, _ = _delta_theory(a_bl, b_bl, P_C)
    pwr_above = [_power_at(n, P_C, a_ab, b_ab) for n in N_RIGHT]
    pwr_below = [_power_at(n, P_C, a_bl, b_bl) for n in N_RIGHT]
    pwr_null  = [_power_at(n, P_C, a_nu, b_nu)  for n in N_RIGHT]

    # draw
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    ax = axes[0]
    n_arr = np.array(N_CONV)
    for i in range(len(PHI_CONV_L2)):
        mu = mean_T_conv[i]; sd = std_T_conv[i]
        ax.fill_between(n_arr, mu - sd, mu + sd, color=C3_CONV[i], alpha=0.15)
        ax.plot(n_arr, mu, color=C3_CONV[i], lw=2.0,
                marker='o', markersize=5, label=phi_labels_conv[i])
        ax.axhline(delta_conv[i], color=C3_CONV[i], lw=1.0, ls=':', alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('$n$', fontsize=13)
    ax.set_ylabel(r'Mean of $T$  ($\pm 1$ SD shaded)', fontsize=13)
    ax.set_title(f'Concentration of $T$  ($p={P_A}$)', fontsize=14)
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1]
    phi_deg = np.degrees(PHI_GRID)
    for i, n in enumerate(N_POWER_B):
        ax.plot(phi_deg, power_phi[i], color=C3_N[i], lw=2.0,
                ls=LS3[i], marker='o', markersize=4, label=f'$n={n}$')
    if not np.isnan(phi_thr):
        ax.axvline(np.degrees(phi_thr),       color='grey', lw=1.0, ls=':')
        ax.axvline(180 - np.degrees(phi_thr), color='grey', lw=1.0, ls=':')
    ax.axhline(alpha, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel(r'Population angle $\varphi_0$ (degrees)', fontsize=13)
    ax.set_ylabel('Empirical power', fontsize=13)
    ax.set_title(f'Power vs $\\varphi_0$  ($p={P_B}$)', fontsize=14)
    ax.set_xlim(0, 180); ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[2]
    lbl_ab = (rf'$|\delta|={d_above:.1f}>t_{{\rm crit}}$'
              rf'  ($\varphi_0={np.degrees(PHI_ABOVE_L2):.0f}°$)')
    lbl_bl = (rf'$|\delta|={d_below:.1f}<t_{{\rm crit}}$'
              rf'  ($\varphi_0={np.degrees(PHI_BELOW_L2):.0f}°$)')
    lbl_nu = r'$H_0$  ($\delta=0$)'
    ax.plot(N_RIGHT, pwr_above, color=C3_RIGHT[0], lw=2.0, ls='-',
            marker='o', markersize=5, label=lbl_ab)
    ax.plot(N_RIGHT, pwr_below, color=C3_RIGHT[1], lw=2.0, ls='--',
            marker='s', markersize=5, label=lbl_bl)
    ax.plot(N_RIGHT, pwr_null,  color=C3_RIGHT[2], lw=2.0, ls=':',
            marker='^', markersize=5, label=lbl_nu)
    ax.axhline(alpha, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('$n$ (log scale)', fontsize=13)
    ax.set_ylabel('Empirical power', fontsize=13)
    ax.set_title(
        f'Power vs $n$  ($p={P_C}$, $t_{{\\rm crit}}={t_crit_c:.2f}$)', fontsize=14)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nFigure fig:power saved to {save_path}")


# =============================================================================
# 9d. Figure fig:power_dual — three-panel dual power validation
# =============================================================================

def figure_power_dual(n_sims=1000, alpha=0.05, seed=0,
                      save_path='power_figure_dual.pdf'):
    """Reproduce Figure 3 (fig:power_dual): three-panel dual distribution validation.

    Each panel shows the empirical distribution of T~ at a fixed phi0 and n=40,
    for increasing p in {40, 80, 160, 1000}.  The theoretical approximation
    t(n-2, delta~) is overlaid as a smooth curve.  As p grows the empirical
    density converges to the theoretical approximation.

    Also shows centre panel: power vs phi0 at p in {40, 80, 160}.
    Also shows right panel: power vs p confirming saturation at pi_inf.

    DGP: dense random unit vectors with angle phi0, effective SNR = 0.5*sqrt(p),
    n=40. The sqrt(p) rescaling keeps the total signal amplitude constant as p
    grows, ensuring the population angle phi0 remains fixed as required by the
    dense-signal assumption of Proposition 5(v).
    """
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure_power_dual."); return

    from scipy.stats import t as t_dist, nct

    rng_dir = np.random.default_rng(seed + 777)
    rng_dat = np.random.default_rng(seed)
    snr = 0.5; n_fixed = 40; df_main = n_fixed - 2
    # Effective SNR is snr * sqrt(p), applied per simulation in _sim below.
    # This keeps the total signal amplitude constant as p grows.
    t_crit_v = t_dist.ppf(1.0 - alpha / 2, df=df_main)
    C4 = ['#2166AC', '#D6604D', '#4DAF4A', '#984EA3']
    C3 = ['#2166AC', '#D6604D', '#4DAF4A']
    LS4 = ['-', '--', ':', '-.']; LS3 = ['-', '--', ':']
    MK4 = ['o', 's', '^', 'D']; MK3 = ['o', 's', '^']

    def _dual_T(X, A, Y):
        # Dual cosine statistic — delegates to cosine_test() dual branch,
        # which uses the corrected Euclidean cosine of (KZ^+ Z a, KZ^+ Z z).
        return cosine_test(X, A, Y)['T']

    def _make_dirs(p, phi0):
        al=rng_dir.standard_normal(p); al/=np.linalg.norm(al)
        perp=rng_dir.standard_normal(p); perp-=(perp@al)*al; perp/=np.linalg.norm(perp)
        return al, np.cos(phi0)*al+np.sin(phi0)*perp

    def _sim(n, p, phi0):
        al,be=_make_dirs(p,phi0); Ts=[]
        eff_snr = snr * np.sqrt(p)   # sqrt(p) rescaling: keeps total signal constant
        for _ in range(n_sims):
            A=rng_dat.standard_normal(n)
            X=eff_snr*np.outer(A,al)+rng_dat.standard_normal((n,p))
            Y=eff_snr*(X@be)+rng_dat.standard_normal(n)
            Ts.append(_dual_T(X,A,Y))
        return np.array(Ts)

    def _power(n, p, phi0):
        return float(np.mean(np.abs(_sim(n,p,phi0)) >= t_crit_v))

    def _delta(phi0):
        return np.cos(phi0)/np.sin(phi0)*np.sqrt(df_main) if abs(np.sin(phi0))>1e-12 else np.inf

    def _pi_inf(phi0):
        d=_delta(phi0)
        if not np.isfinite(d): return 1.0
        if abs(d)<1e-12: return alpha
        return 1-nct.cdf(t_crit_v,df=df_main,nc=abs(d))+nct.cdf(-t_crit_v,df=df_main,nc=abs(d))

    # Left panel: density at phi0=60, p in {40, 80, 160, 1000}
    phi0_den  = np.radians(60)   # changed from 70 to 60
    delta_den = _delta(phi0_den)
    p_panels  = [40, 80, 160, 1000]
    p_labels  = [f'$p=40$', f'$p=80$', f'$p=160$', f'$p=1000$']
    T_panels  = []
    for p in p_panels:
        print(f"  Density panel p={p} ...")
        T_panels.append(_sim(n_fixed, p, phi0_den))

    # Centre panel: power vs phi0 at p=40,80,160 (changed from 40,50,60)
    print("  Panel 2: power vs phi0 ...")
    p_list    = [40, 80, 160]
    phi_grid  = np.linspace(0.08, np.pi-0.08, 25)
    power_phi = np.zeros((3, len(phi_grid)))
    for pi_idx, p in enumerate(p_list):
        for fi, phi0 in enumerate(phi_grid):
            power_phi[pi_idx,fi] = _power(n_fixed,p,phi0)
        print(f"    p={p} done")
    phi_fine     = np.linspace(0.01, np.pi-0.01, 200)
    pi_inf_curve = np.array([_pi_inf(ph) for ph in phi_fine])

    # Panel 3: power vs p at four angles
    print("  Panel 3: power vs p ...")
    angles3     = [np.radians(60), np.radians(70), np.radians(80), np.radians(90)]
    pi_inf_vals = [_pi_inf(ph) for ph in angles3]
    labels3 = [
        rf'$\varphi_0=60°$  ($\pi_\infty={pi_inf_vals[0]:.2f}$)',
        rf'$\varphi_0=70°$  ($\pi_\infty={pi_inf_vals[1]:.2f}$)',
        rf'$\varphi_0=80°$  ($\pi_\infty={pi_inf_vals[2]:.2f}$)',
        rf'$\varphi_0=90°$  ($\pi_\infty=\alpha={alpha}$)',
    ]
    p_grid3 = [44, 100, 200, 400, 800, 1600]
    power_p = np.zeros((4, len(p_grid3)))
    for j, p in enumerate(p_grid3):
        for i, phi0 in enumerate(angles3):
            power_p[i,j] = _power(n_fixed,p,phi0)
        print(f"    p={p} done")

    # draw: three panels
    from scipy.stats import gaussian_kde
    t_grid  = np.linspace(-5, 12, 600)
    nct_pdf = nct.pdf(t_grid, df=df_main, nc=delta_den)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))

    # Left: density comparison
    ax = axes[0]
    C_EMP = ['#2166AC', '#D6604D', '#4DAF4A', '#984EA3']
    for k, (Ts, plabel, col) in enumerate(zip(T_panels, p_labels, C_EMP)):
        kde = gaussian_kde(Ts, bw_method='scott')
        ax.fill_between(t_grid, kde(t_grid), alpha=0.18, color=col)
        ax.plot(t_grid, kde(t_grid), color=col, lw=2.0, label=plabel)
    ax.plot(t_grid, nct_pdf, color='k', lw=2.5, ls='--',
            label=rf'$t(38,\delta)$, $\delta={delta_den:.2f}$')
    ax.set_xlabel(r'$T$', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(
        rf'Density convergence ($n={n_fixed}$, $\varphi_0=60°$)',
        fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    ax.set_xlim(-3, 12); ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Centre: power vs phi0
    ax = axes[1]
    for pi_idx, p in enumerate(p_list):
        ax.plot(np.degrees(phi_grid), power_phi[pi_idx], color=C3[pi_idx], lw=2.0,
                ls=LS3[pi_idx], marker='o', markersize=4, label=f'$p={p}$')
    ax.plot(np.degrees(phi_fine), pi_inf_curve, color='k', lw=1.5, ls='--', alpha=0.7,
            label=r'$\pi_\infty$ asymptote')
    ax.axhline(alpha, color='k', lw=0.6, ls=':', alpha=0.4)
    ax.set_xlabel(r'$\varphi_0$ (degrees)', fontsize=13)
    ax.set_ylabel('Empirical power', fontsize=13)
    ax.set_title(f'Power vs $\\varphi_0$  ($n={n_fixed}$)', fontsize=13)
    ax.set_xlim(0, 180); ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=9, frameon=False); ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Right: power vs p
    ax = axes[2]
    for i in range(4):
        ax.plot(p_grid3, power_p[i], color=C4[i], lw=2.0, ls=LS4[i],
                marker=MK4[i], markersize=5, label=labels3[i])
        ax.axhline(pi_inf_vals[i], color=C4[i], lw=1.0, ls=':', alpha=0.7,
                   label=(r'$\pi_\infty$ asymptote' if i == 0 else '_nolegend_'))
    ax.set_xscale('log')
    ax.set_xlabel('$p$ (number of mediators)', fontsize=13)
    ax.set_ylabel('Empirical power', fontsize=13)
    ax.set_title(f'Power vs $p$  ($n={n_fixed}$)', fontsize=13)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=9, frameon=False); ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nFigure fig:power_dual saved to {save_path}")


# =============================================================================
# 9e. Figure fig:null_dist — QQ plots under three null scenarios
# =============================================================================

def figure_null_dist(n_datasets=1000, seed=0, save_path='null_dist_fig.pdf'):
    """Reproduce Figure 1 (fig:null_dist): two-panel QQ plot under H0.

    Left panel  — primal (n=1000, p=100): T  vs t(p-1) = t(99).
    Right panel — dual   (n=100, p=1000): T~ vs t(n-2) = t(98).

    Three null scenarios overlaid in each panel (n_datasets replicates each):
      Blue  : both paths inactive
      Red   : alpha-path only active (A correlated with X[:,j])
      Green : beta-path  only active (Y correlated with X[:,j])

    AR(1) DGP: rho=0.75, sigma=0.5, tau=0.  Dual panel uses fast AR(1) recursion
    to avoid the O(p^2) Cholesky at p=1000.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure_null_dist.")
        return

    from scipy.stats import t as t_dist

    rng = np.random.default_rng(seed)
    scenario_colors = ['#4878CF', '#E8886A', '#6ACC65']
    null_labels     = ['Both inactive', r'$\alpha$-path active', r'$\beta$-path active']

    def _make_X(n, p, rho=0.75):
        # Use AR(1) recursion for large p to avoid O(p^2) Cholesky.
        eps = rng.standard_normal((n, p))
        if p == 1:
            return eps - eps.mean(0)
        X = np.empty((n, p))
        X[:, 0] = eps[:, 0]
        scale = np.sqrt(1.0 - rho**2)
        for j in range(1, p):
            X[:, j] = rho * X[:, j-1] + scale * eps[:, j]
        return X - X.mean(0)

    def _simulate(n, p):
        """Simulate n_datasets replicates for each of the 3 null scenarios."""
        results = []
        for ni in range(3):
            Ts = []
            for _ in range(n_datasets):
                X = _make_X(n, p)
                A = 0.5 * rng.standard_normal(n); A -= A.mean()
                Y = 0.5 * rng.standard_normal(n); Y -= Y.mean()
                if ni == 1:
                    idx = rng.integers(p)
                    A = X[:, idx] + 0.5 * rng.standard_normal(n); A -= A.mean()
                elif ni == 2:
                    idx = rng.integers(p)
                    Y = X[:, idx] + 0.5 * rng.standard_normal(n); Y -= Y.mean()
                Ts.append(cosine_test(X, A, Y)['T'])
            results.append(np.sort(Ts))
        return results

    print("  Simulating primal (n=1000, p=100) ...")
    T_primal = _simulate(n=1000, p=100)   # T  ~ t(99) under H0
    print("  Simulating dual   (n=100, p=1000) ...")
    T_dual   = _simulate(n=100,  p=1000)  # T~ ~ t(98) under H0

    probs       = (np.arange(1, n_datasets + 1) - 0.5) / n_datasets
    theo_primal = t_dist.ppf(probs, df=99)
    theo_dual   = t_dist.ppf(probs, df=98)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    panel_cfg = [
        (axes[0], T_primal, theo_primal,
         r'Theoretical $t(p{-}1)=t(99)$ quantile',
         r'Primal: $n=1000,\ p=100$'),
        (axes[1], T_dual,   theo_dual,
         r'Theoretical $t(n{-}2)=t(98)$ quantile',
         r'Dual: $n=100,\ p=1000$'),
    ]
    for ax, T_list, theo, xlabel, title in panel_cfg:
        for ni in range(3):
            ax.scatter(theo, T_list[ni], s=18, alpha=0.60,
                       color=scenario_colors[ni], linewidths=0)
        all_emp = np.concatenate(T_list)
        lo  = min(theo[0],  all_emp.min())
        hi  = max(theo[-1], all_emp.max())
        pad = (hi - lo) * 0.03
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], 'k-', lw=1.2, zorder=5)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel('Empirical quantile', fontsize=13)
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=scenario_colors[ni],
                   markersize=11, label=null_labels[ni])
        for ni in range(3)
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nFigure fig:null_dist saved to {save_path}")


# =============================================================================
# 9e. Table tab:inference — empirical size and power (cosine test vs IUT)
# =============================================================================

def table_inference(n=200, p_vals=(20, 40, 80), n_null=1000, n_power=1000,
                    signals=(0.10, 0.20), alpha=0.05, seed=0):
    """Reproduce Table tab:inference: empirical rejection rates at alpha=0.05.

    Top panel (primal): n=200, p in {20,40,80}, AR(1) DGP (rho=0.75, tau=0.25,
    sigma=0.5).  Three null scenarios (type I error) and two signal levels (power),
    comparing cosine test against the IUT.

    Bottom panel (dual): p=200, n in {20,40,60,80,100,120}, iid X ~ N(0,I_p),
    tau=0, sigma=0.5.  Same null/power structure; IUT not applicable.
    """
    from scipy.stats import f as f_dist

    rng = np.random.default_rng(seed)

    # ── Top panel helpers (AR(1), tau=0.25) ───────────────────────────────────
    def _make_dataset_top(n, p, ni=None, sig=None):
        """Generate one dataset for the top panel under null scenario ni
        (ni=0: both inactive, ni=1: alpha-path, ni=2: beta-path)
        or under H1 with signal magnitude sig."""
        idx = np.arange(p)
        Sig = 0.75 ** np.abs(idx[:, None] - idx[None, :])
        X   = rng.standard_normal((n, p)) @ np.linalg.cholesky(Sig).T
        X  -= X.mean(0)
        if sig is not None:
            # Power: both paths share single nonzero entry at random index
            j = rng.integers(p)
            A = sig * X[:, j] + 0.5 * rng.standard_normal(n)
            Y = sig * X[:, j] + 0.25 * A + 0.5 * rng.standard_normal(n)
        else:
            # Type I error
            if ni == 0:   # both inactive
                A = 0.5 * rng.standard_normal(n)
                Y = 0.25 * A + 0.5 * rng.standard_normal(n)
            elif ni == 1: # alpha-path only (A correlated with X)
                j = rng.integers(p)
                A = X[:, j] + 0.5 * rng.standard_normal(n)
                Y = 0.25 * A + 0.5 * rng.standard_normal(n)
            else:         # beta-path only (Y correlated with X)
                j = rng.integers(p)
                A = 0.5 * rng.standard_normal(n)
                Y = X[:, j] + 0.25 * A + 0.5 * rng.standard_normal(n)
        A -= A.mean(); Y -= Y.mean()
        return X, A, Y

    # ── Bottom panel helpers (iid X, tau=0) ───────────────────────────────────
    def _make_dataset_bot(n, p, ni, sig=None):
        X  = rng.standard_normal((n, p)); X -= X.mean(0)
        if sig is not None:
            j = rng.integers(p)
            A = sig * X[:, j] + 0.5 * rng.standard_normal(n)
            Y = sig * X[:, j] + 0.5 * rng.standard_normal(n)
        else:
            if ni == 0:
                A = 0.5 * rng.standard_normal(n)
                Y = 0.5 * rng.standard_normal(n)
            elif ni == 1:
                j = rng.integers(p)
                A = X[:, j] + 0.5 * rng.standard_normal(n)
                Y = 0.5 * rng.standard_normal(n)
            else:
                j = rng.integers(p)
                A = 0.5 * rng.standard_normal(n)
                Y = X[:, j] + 0.5 * rng.standard_normal(n)
        A -= A.mean(); Y -= Y.mean()
        return X, A, Y

    def _iut_reject(X, A, Y):
        n, p = X.shape
        Xc = X - X.mean(0); Ac = A - A.mean(); Yc = Y - Y.mean()
        Z  = Xc - np.outer(Ac, (Ac @ Xc) / (Ac @ Ac))
        def _f_pval(design, response):
            coef   = np.linalg.lstsq(design, response, rcond=None)[0]
            ss_res = float(response @ response) - float(coef @ (design.T @ response))
            ss_tot = float(response @ response)
            r2     = max(1.0 - ss_res / (ss_tot + 1e-15), 0.0)
            F      = (r2 / p) / ((1 - r2) / (n - p - 1) + 1e-15)
            return float(f_dist.sf(F, dfn=p, dfd=n - p - 1))
        return _f_pval(Xc, Ac) < alpha and _f_pval(Z, Yc) < alpha

    null_labels = ['Both inactive      ',
                   'alpha-path active  ',
                   'beta-path active   ']
    cw = 8

    # ── Top panel ─────────────────────────────────────────────────────────────
    hdr = f"{'':30}" + "".join(f"{'p='+str(p):>{2*cw}}" for p in p_vals)
    sub = f"{'':30}" + "".join(f"{'Cosine':>{cw}}{'IUT':>{cw}}" for _ in p_vals)
    print(f"\nTable tab:inference (top panel): AR(1) DGP, n={n}, tau=0.25, "
          f"alpha={alpha}, {n_null} datasets per cell")
    print(hdr); print(sub); print("-" * len(sub))

    print("Type I error (nominal 0.05)")
    for ni, label in enumerate(null_labels):
        row = f"  {label:<28}"
        for p in p_vals:
            rej_cos = rej_iut = 0
            for _ in range(n_null):
                X, A, Y = _make_dataset_top(n, p, ni)
                if cosine_test(X, A, Y)['p_value_two_sided'] < alpha: rej_cos += 1
                if _iut_reject(X, A, Y):                               rej_iut += 1
            row += f"{rej_cos/n_null:{cw}.3f}{rej_iut/n_null:{cw}.3f}"
        print(row)

    print("\nPower (both paths active, shared nonzero entry)")
    for sig in signals:
        row = f"  Signal = {sig:.2f}              "
        for p in p_vals:
            rej_cos = rej_iut = 0
            for _ in range(n_power):
                X, A, Y = _make_dataset_top(n, p, sig=sig)
                if cosine_test(X, A, Y)['p_value_two_sided'] < alpha: rej_cos += 1
                if _iut_reject(X, A, Y):                               rej_iut += 1
            row += f"{rej_cos/n_power:{cw}.3f}{rej_iut/n_power:{cw}.3f}"
        print(row)

    # ── Bottom panel (dual) ───────────────────────────────────────────────────
    p_dual  = 200
    n_vals  = [20, 40, 60, 80, 100, 120]
    signals_dual = [0.30, 0.50]
    cw2 = 7
    hdr2 = f"{'':30}" + "".join(f"{'n='+str(nv):>{cw2}}" for nv in n_vals)
    print(f"\nTable tab:inference (bottom panel): iid X, p={p_dual}, tau=0, "
          f"alpha={alpha}, {n_null} datasets per cell")
    print(hdr2); print("-" * len(hdr2))

    print("Type I error (nominal 0.05)")
    for ni, label in enumerate(null_labels):
        row = f"  {label:<28}"
        for nv in n_vals:
            rej_cos = 0
            for _ in range(n_null):
                X, A, Y = _make_dataset_bot(nv, p_dual, ni)
                if cosine_test(X, A, Y)['p_value_two_sided'] < alpha: rej_cos += 1
            row += f"{rej_cos/n_null:{cw2}.3f}"
        print(row)

    print("\nPower (both paths active, shared nonzero entry)")
    for sig in signals_dual:
        row = f"  Signal = {sig:.2f}              "
        for nv in n_vals:
            rej_cos = 0
            for _ in range(n_power):
                X, A, Y = _make_dataset_bot(nv, p_dual, ni=None, sig=sig)
                if cosine_test(X, A, Y)['p_value_two_sided'] < alpha: rej_cos += 1
            row += f"{rej_cos/n_power:{cw2}.3f}"
        print(row)


# =============================================================================
# 10. Master runner — reproduces all paper tables and figures
# =============================================================================

def run_all():
    """Reproduce all simulation outputs in the paper, in order:
        tab:obj_h     -- Table 1: mean indirect effect h
        tab:obj_f     -- Table 2: mean mediation index f*
        tab:time      -- Table 3: timing vs numerical baselines
        fig:power     -- Figure 2: three-panel power validation
        fig:null_dist -- Figure 1: QQ plots under H0
        tab:inference -- Table 4: size and power of cosine test vs IUT
    """
    print("=" * 70)
    print("optmed.py — Reproducing all paper tables and figures")
    print("=" * 70)

    print("\n--- Tables tab:obj_h and tab:obj_f (share one simulation pass) ---")
    cache = table_obj_h(n_seeds=20)
    table_obj_f(cache=cache)

    print("\n--- Table tab:time ---")
    table_time()

    print("\n--- Figure fig:power (Fig 2) ---")
    figure_power_props(n_sims=1000, save_path='power_figure.pdf')

    print("\n--- Figure fig:power_dual (Fig 3) ---")
    figure_power_dual(n_sims=1000, save_path='power_figure_dual.pdf')

    print("\n--- Figure fig:null_dist (Fig 1) ---")
    figure_null_dist(n_datasets=1000, save_path='null_dist_fig.pdf')

    print("\n--- Table tab:inference ---")
    table_inference(n_null=1000, n_power=1000)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_all()

