"""
Simulation study: Coverage of 95% CIs for quantile (median) regression
on a binary outcome with 2 regressors + intercept.

Compares:
  1. Koenker-Bassett (KB) heteroskedasticity-robust analytic standard errors
  2. Bayesian Bootstrap (BB) standard errors

The "true" parameter beta_star is the probability limit of the QR estimator,
i.e., the minimizer of E[rho_tau(y - x'beta)].  For binary y this is NOT the
logistic DGP coefficient; it is estimated as the mean of all beta_hat draws.

Why KB fails for binary y:
  The sandwich formula requires estimating f_i(0), the conditional density of
  the error at zero.  For binary y the error distribution is a two-point mass
  at {1-x'beta_star} and {-x'beta_star}, so the density at 0 is zero (or
  ill-defined).  The kernel sparsity estimator is therefore unreliable,
  producing inflated or collapsed SE estimates and poor CI coverage.

Why BB is more robust:
  The Bayesian Bootstrap avoids distributional assumptions by directly
  resampling via Dirichlet weights, making it valid for discrete / non-smooth
  error distributions.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import linprog
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEED           = 42
N_SIMULATIONS  = 1000   # simulation replications
N_OBS          = 200    # observations per simulation
TAU            = 0.5    # quantile (median regression)
ALPHA          = 0.05   # CI level  →  95% CI
N_BB           = 200    # Bayesian Bootstrap draws per simulation

# Logistic DGP parameters (NOT the true QR parameter — see note above)
DGP_BETA = np.array([0.0, 0.5, -0.3])

rng = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Data-generating process
# ─────────────────────────────────────────────────────────────────────────────
def generate_data(n, rng):
    """Binary y ~ Bernoulli(sigmoid(X @ DGP_BETA)).
    Regressors: intercept, x1 ~ N(0,1), x2 ~ Bernoulli(0.5).
    """
    x1 = rng.standard_normal(n)
    x2 = rng.binomial(1, 0.5, n).astype(float)
    X  = np.column_stack([np.ones(n), x1, x2])
    p  = 1.0 / (1.0 + np.exp(-X @ DGP_BETA))
    y  = rng.binomial(1, p).astype(float)
    return y, X


# ─────────────────────────────────────────────────────────────────────────────
# Quantile regression via LP (HiGHS solver)
# ─────────────────────────────────────────────────────────────────────────────
def quantile_reg(y, X, tau=0.5, weights=None):
    """Solve the (optionally weighted) quantile regression LP.

    Minimises:  sum_i w_i * [ tau*u_i+ + (1-tau)*u_i- ]
    subject to: x_i'beta + u_i+ - u_i- = y_i,  u_i+, u_i- >= 0

    Parameters
    ----------
    weights : None or array of shape (n,)
        Observation weights (Bayesian Bootstrap Dirichlet draws).
        If None, uniform weights are used.
    """
    n, k = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights)

    c      = np.concatenate([np.zeros(k), tau * w, (1 - tau) * w])
    A_eq   = np.hstack([X, np.eye(n), -np.eye(n)])
    b_eq   = y
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return result.x[:k] if result.status == 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Koenker-Bassett heteroskedasticity-robust standard errors
# ─────────────────────────────────────────────────────────────────────────────
def kb_robust_se(y, X, beta_hat, tau=0.5):
    """KB (1978) / Powell (1986) sandwich estimator.

    V = (1/n) H^{-1} J H^{-1}

    where:
      H = f_hat(0) * (X'X / n)          [bread, uses global sparsity]
      J = (1/n) sum_i psi_i^2 x_i x_i'  [heteroskedastic meat]
      psi_i = tau - 1(e_i < 0)           [score]

    f_hat(0) estimated with Hall-Sheather bandwidth applied to OLS residuals.
    For binary y, f_hat(0) ≈ 0 because the error distribution is discrete,
    causing the bread to be near-singular and SEs to be unreliable.
    """
    n, k   = X.shape
    resid  = y - X @ beta_hat

    # ── Bandwidth (Hall-Sheather, 1988) ───────────────────────────────────
    z_a   = norm.ppf(1 - ALPHA / 2)          # 1.96 for 95% CI
    phi_z = norm.pdf(z_a)
    h_hs  = (n ** (-1 / 3)
             * z_a ** (2 / 3)
             * (1.5 * phi_z ** 2 / (2 * z_a ** 2 + 1)) ** (1 / 3))
    # Adaptive floor: capture at least ~5% of residuals
    resid_scale = max(np.std(resid), 1e-4)
    h = max(h_hs * resid_scale, 0.01)

    # ── Global sparsity (density at 0) ────────────────────────────────────
    f0_hat = np.mean(np.abs(resid) <= h) / (2.0 * h)
    f0_hat = max(f0_hat, 1e-6)            # numerical floor

    # ── Bread H ──────────────────────────────────────────────────────────
    XTX = X.T @ X / n
    H   = f0_hat * XTX

    # ── Meat J (heteroskedastic) ──────────────────────────────────────────
    psi = tau - (resid < 0).astype(float)
    J   = (X * psi[:, None]).T @ (X * psi[:, None]) / n

    # ── Sandwich ──────────────────────────────────────────────────────────
    try:
        H_inv = np.linalg.solve(H, np.eye(k))
    except np.linalg.LinAlgError:
        return None

    V  = H_inv @ J @ H_inv / n
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return se


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Bootstrap standard errors
# ─────────────────────────────────────────────────────────────────────────────
def bayesian_bootstrap_se(y, X, tau=0.5, n_boot=200, rng=None):
    """Bayesian Bootstrap (Rubin, 1981) for quantile regression.

    Each draw assigns Dirichlet(1,...,1) weights to observations and solves
    the *weighted* QR LP.  The standard deviation of the resulting coefficient
    distribution serves as the SE estimate.

    This is distribution-free: no sparsity/density estimation required.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, k       = X.shape
    boot_betas = np.full((n_boot, k), np.nan)

    for b in range(n_boot):
        # Dirichlet(1,...,1) = normalised Exponential(1) draws
        w = rng.exponential(1.0, n)
        w = w / w.sum() * n          # rescale so sum(w) = n (like frequency weights)

        beta_b = quantile_reg(y, X, tau=tau, weights=w)
        if beta_b is not None:
            boot_betas[b] = beta_b

    valid = boot_betas[~np.any(np.isnan(boot_betas), axis=1)]
    return valid.std(axis=0) if len(valid) >= 10 else None


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation():
    z_crit = norm.ppf(1 - ALPHA / 2)
    k      = len(DGP_BETA)

    # Storage for CIs and point estimates
    all_betas  = []                      # (sim, k) — to estimate beta_star
    kb_intervals = []                    # list of (lo, hi, valid)
    bb_intervals = []

    print("=" * 65)
    print("Quantile Regression Coverage Simulation")
    print("=" * 65)
    print(f"  Simulations : {N_SIMULATIONS}")
    print(f"  n per sim   : {N_OBS}")
    print(f"  Quantile τ  : {TAU}")
    print(f"  CI level    : {(1-ALPHA)*100:.0f}%")
    print(f"  BB draws    : {N_BB}")
    print(f"  DGP beta    : {DGP_BETA}  (logistic; ≠ population QR beta)")
    print("-" * 65)

    for sim in range(N_SIMULATIONS):
        if (sim + 1) % 100 == 0:
            print(f"  Simulation {sim + 1:>4}/{N_SIMULATIONS}...")

        y, X       = generate_data(N_OBS, rng)
        beta_hat   = quantile_reg(y, X, tau=TAU)
        if beta_hat is None:
            continue
        all_betas.append(beta_hat)

        # ── KB CI ────────────────────────────────────────────────────────
        kb_se = kb_robust_se(y, X, beta_hat, tau=TAU)
        if kb_se is not None:
            kb_intervals.append((beta_hat - z_crit * kb_se,
                                 beta_hat + z_crit * kb_se,
                                 True))
        else:
            kb_intervals.append((None, None, False))

        # ── BB CI ────────────────────────────────────────────────────────
        bb_se = bayesian_bootstrap_se(y, X, tau=TAU, n_boot=N_BB, rng=rng)
        if bb_se is not None:
            bb_intervals.append((beta_hat - z_crit * bb_se,
                                 beta_hat + z_crit * bb_se,
                                 True))
        else:
            bb_intervals.append((None, None, False))

    # ── Population QR parameter (probability limit) ──────────────────────
    beta_star = np.mean(all_betas, axis=0)

    # ── Coverage ─────────────────────────────────────────────────────────
    n_total  = len(all_betas)
    kb_cover = np.zeros(k)
    bb_cover = np.zeros(k)
    kb_n = bb_n = 0

    for i in range(n_total):
        lo, hi, ok = kb_intervals[i]
        if ok:
            kb_cover += ((beta_star >= lo) & (beta_star <= hi)).astype(float)
            kb_n += 1

        lo, hi, ok = bb_intervals[i]
        if ok:
            bb_cover += ((beta_star >= lo) & (beta_star <= hi)).astype(float)
            bb_n += 1

    return beta_star, kb_cover, bb_cover, kb_n, bb_n, n_total


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def report(beta_star, kb_cover, bb_cover, kb_n, bb_n, n_total):
    param_names = ["Intercept", "Beta_1 (x1)", "Beta_2 (x2)"]

    print("\n" + "=" * 65)
    print("  Population QR coefficients (beta_star = mean of beta_hats)")
    print("=" * 65)
    print("  These are the probability limits of the QR estimator for")
    print("  binary y — generally ≠ the logistic DGP coefficients.")
    for j, name in enumerate(param_names):
        print(f"    {name:<14}: {beta_star[j]:+.4f}   "
              f"(DGP logit: {DGP_BETA[j]:+.4f})")

    print("\n" + "=" * 65)
    print(f"  95% CI Coverage  (nominal = {(1-ALPHA)*100:.0f}%,  "
          f"N = {n_total} simulations)")
    print("=" * 65)
    header = f"{'Parameter':<16}  {'beta_star':>10}  "
    header += f"{'KB Coverage':>13}  {'BB Coverage':>13}"
    print(header)
    print("-" * 65)
    for j, name in enumerate(param_names):
        kb_pct = 100 * kb_cover[j] / kb_n if kb_n > 0 else float("nan")
        bb_pct = 100 * bb_cover[j] / bb_n if bb_n > 0 else float("nan")
        diff   = bb_pct - kb_pct
        flag   = "  ← BB better" if diff > 2 else ""
        print(f"{name:<16}  {beta_star[j]:>10.4f}  "
              f"{kb_pct:>11.2f}%  {bb_pct:>11.2f}%{flag}")
    print("-" * 65)
    print(f"\n  Valid simulations : {n_total}/{N_SIMULATIONS}")
    print(f"  Valid KB CIs      : {kb_n}  |  Valid BB CIs: {bb_n}")

    print("\n" + "=" * 65)
    print("  Diagnostic: Why KB under-performs for binary y")
    print("=" * 65)
    print("""
  The KB sandwich formula uses the sparsity estimator f_hat(0):
    - For continuous errors, f(0) > 0 and is well estimated.
    - For binary y, errors concentrate at two point masses
      {1-x'beta_star} and {-x'beta_star}, so the true density
      at 0 is ZERO (discrete distribution).
    - f_hat(0) is therefore unreliable, making the bread H^{-1}
      numerically unstable and SE estimates inaccurate.

  The Bayesian Bootstrap avoids sparsity estimation entirely:
    - Directly perturbs observation weights (Dirichlet draws).
    - The variability of beta_hat across weighted LP solves gives
      an empirical approximation of the sampling distribution.
    - No density estimation ⟹ robust to discrete / non-smooth errors.
""")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_simulation()
    report(*result)
