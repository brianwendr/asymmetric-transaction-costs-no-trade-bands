"""
engine.py -- core backtest and inference routines.

Model (two assets: equity ETF S and bond ETF B; w = equity weight).
Each trading day t:
    pre-trade weight   w_pre = w R_S / (w R_S + (1-w) R_B)
    on a monitoring day the policy maps w_pre -> w_new
    trade cost (fraction of wealth)
        c_up(t) * (w_new - w_pre)    if w_new > w_pre   (buy S, sell B)
        c_dn(t) * (w_pre - w_new)    if w_new < w_pre   (sell S, buy B)
    with c_up = c_S^buy + c_B^sell  and  c_dn = c_S^sell + c_B^buy.

Policies are encoded by (monitor mask, lower bound L, upper bound U, mode):
    mode 0 ("boundary"): w_new = clip(w_pre, L, U)          -> no-trade band / play operator
    mode 1 ("target")  : w_new = w_star if w_pre outside [L,U], else w_pre
Calendar rebalancing is mode 1 with L = U = w_star on calendar dates;
buy-and-hold has an empty monitor mask.

All configurations (and bootstrap paths) are simulated simultaneously:
arrays have a leading dimension K = number of (path x configuration) pairs.
"""
import numpy as np

TD = 252.0


# ---------------------------------------------------------------- simulation
def simulate(RS, RB, monitor, L, U, mode, wstar, cup, cdn, w0=None, keep="all"):
    """
    RS, RB  : (T,) or (K,T) gross returns
    monitor : (K,T) bool
    L, U    : (K,) or (K,T) bounds
    mode    : (K,) int {0,1}
    wstar   : (K,) target weight
    cup,cdn : (T,) or (K,T) one-way round-trip cost rates (decimal)
    returns dict of (K,T) arrays: net return r, pre-trade weight wpre,
            post-trade weight w, traded |dw|, cost
    """
    K = monitor.shape[0]
    T = monitor.shape[1]
    RS = np.broadcast_to(RS, (K, T)); RB = np.broadcast_to(RB, (K, T))
    L = np.broadcast_to(L if np.ndim(L) == 2 else np.asarray(L)[:, None], (K, T))
    U = np.broadcast_to(U if np.ndim(U) == 2 else np.asarray(U)[:, None], (K, T))
    cup = np.broadcast_to(cup, (K, T)); cdn = np.broadcast_to(cdn, (K, T))
    mode = np.asarray(mode); wstar = np.asarray(wstar, float)
    w = wstar.copy() if w0 is None else np.asarray(w0, float).copy()
    full = keep == "all"
    r = np.empty((K, T)); cst = np.empty((K, T))
    if full:
        wp = np.empty((K, T)); wo = np.empty((K, T)); trd = np.empty((K, T))
    is_tgt = mode == 1
    for t in range(T):
        gs = w * RS[:, t]; gb = (1.0 - w) * RB[:, t]; g = gs + gb
        wpre = gs / g
        lo = L[:, t]; hi = U[:, t]
        wb = np.minimum(np.maximum(wpre, lo), hi)                 # boundary rule
        outside = (wpre < lo) | (wpre > hi)
        wt = np.where(outside, wstar, wpre)                       # target rule
        wn = np.where(is_tgt, wt, wb)
        wn = np.where(monitor[:, t], wn, wpre)
        dw = wn - wpre
        c = np.where(dw > 0, cup[:, t] * dw, -cdn[:, t] * dw)
        r[:, t] = g * (1.0 - c) - 1.0
        cst[:, t] = c
        if full:
            wp[:, t] = wpre; wo[:, t] = wn; trd[:, t] = np.abs(dw)
        w = wn
    if not full:
        return dict(r=r, cost=cst)
    return dict(r=r, wpre=wp, w=wo, trade=trd, cost=cst)


# ---------------------------------------------------------------- metrics
def max_drawdown(r):
    v = np.cumprod(1.0 + r, axis=-1)
    peak = np.maximum.accumulate(v, axis=-1)
    return (1.0 - v / peak).max(axis=-1)


def metrics(sim, rf, r_target, gamma=5.0):
    """Annualised performance metrics; rf, r_target: (T,) or (K,T)."""
    r = sim["r"]
    ex = r - rf
    mu = r.mean(-1) * TD
    vol = r.std(-1, ddof=1) * np.sqrt(TD)
    exmu = ex.mean(-1) * TD
    sr = exmu / (ex.std(-1, ddof=1) * np.sqrt(TD))
    years = r.shape[-1] / TD
    cagr = np.prod(1 + r, axis=-1) ** (1 / years) - 1
    ce = exmu - 0.5 * gamma * vol ** 2
    te = (r - r_target).std(-1, ddof=1) * np.sqrt(TD)
    turn = sim["trade"].sum(-1) / years
    drag = sim["cost"].sum(-1) / years
    ntr = (sim["trade"] > 1e-12).sum(-1) / years
    return dict(mean=mu, vol=vol, sharpe=sr, cagr=cagr, ce=ce, te=te,
                turnover=turn, costdrag=drag, ntrades=ntr,
                mdd=max_drawdown(r), avgw=sim["w"].mean(-1))


# ---------------------------------------------------------------- bootstrap
def stationary_indices(T, B, mean_block, rng):
    """Politis-Romano (1994) stationary bootstrap index matrix (B,T)."""
    p = 1.0 / mean_block
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    newblk = rng.random((B, T)) < p
    starts = rng.integers(0, T, (B, T))
    for t in range(1, T):
        idx[:, t] = np.where(newblk[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def paired_test(ra, rb, rf, B=10000, mean_block=21, gamma=5.0, seed=1):
    """
    Paired stationary bootstrap for differences between two realised daily
    return series (same dates).  Returns point estimates, 95% percentile CIs
    and two-sided p-values (null-centred) for
        mean difference (annualised), Sharpe difference, CE difference.
    """
    rng = np.random.default_rng(seed)
    T = len(ra)
    ea, eb = ra - rf, rb - rf

    def stats(xa, xb):
        mda = (xa.mean(-1) - xb.mean(-1)) * TD
        sa = xa.mean(-1) / xa.std(-1, ddof=1) * np.sqrt(TD)
        sb = xb.mean(-1) / xb.std(-1, ddof=1) * np.sqrt(TD)
        cea = xa.mean(-1) * TD - 0.5 * gamma * xa.var(-1, ddof=1) * TD
        ceb = xb.mean(-1) * TD - 0.5 * gamma * xb.var(-1, ddof=1) * TD
        return np.stack([mda, sa - sb, cea - ceb], -1)

    est = stats(ea, eb)
    idx = stationary_indices(T, B, mean_block, rng)
    bs = stats(ea[idx], eb[idx])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    cen = bs - bs.mean(0)
    p = (np.abs(cen) >= np.abs(est)).mean(0)
    return dict(est=est, lo=lo, hi=hi, p=p)


# ---------------------------------------------------------------- helpers
def calendar_mask(dates, freq):
    """True on the last trading day of each month ('M'), quarter ('Q') or year ('A')."""
    import pandas as pd
    d = pd.DatetimeIndex(dates)
    if freq == "M":
        key = d.year * 100 + d.month
    elif freq == "Q":
        key = d.year * 10 + (d.month - 1) // 3
    elif freq == "A":
        key = d.year
    else:
        raise ValueError(freq)
    key = np.asarray(key)
    m = np.zeros(len(d), bool)
    m[:-1] = key[1:] != key[:-1]
    m[-1] = True
    return m


def stress_indicator(RS, window=21, q=0.80, burn=252):
    """
    Real-time stress flag: trailing `window`-day realised volatility of the
    equity leg (through day t-1) exceeds the expanding-window q-quantile of its
    own history.  Uses only information available before trading on day t.
    """
    lr = np.log(RS)
    T = len(lr)
    rv = np.full(T, np.nan)
    cs = np.concatenate([[0.0], np.cumsum(lr ** 2)])
    for t in range(window, T):
        rv[t] = np.sqrt((cs[t] - cs[t - window]) / window)      # uses days t-window..t-1
    flag = np.zeros(T, bool)
    hist = []
    for t in range(T):
        if not np.isnan(rv[t]):
            if len(hist) >= burn:
                flag[t] = rv[t] > np.quantile(hist, q)
            hist.append(rv[t])
    return flag
