#!/usr/bin/env python3
"""
02_analysis.py -- produces every number, table and figure in the paper.

Pre-specified design (fixed before results were inspected):
  * Primary sample   : SPY / IEF, w* = 0.60, daily monitoring.
  * Primary friction : sell-side equity surcharge psi = 50 bp on top of a 5 bp
                       one-way cost on every leg (Table 1, Panel A).
  * Statutory case   : Taiwan, 0050 / 00679B; 14.25 bp commission on every leg,
                       0.1% securities transaction tax on equity-ETF sales,
                       bond ETF exempt (Securities Transaction Tax Act, Art. 2-1).
  * Primary metric   : net certainty-equivalent excess return, gamma = 5.
  * Out-of-sample    : band widths re-selected every January by maximising the
                       training-window CE on an expanding window
                       (5-year burn-in for US samples, 3-year for Taiwan).
  * Inference        : paired stationary bootstrap (Politis & Romano, 1994),
                       mean block 21 days, B = 10,000.
  * Mechanism        : re-simulation over 1,000 stationary-bootstrap histories
                       (mean block 63 days) of the two asset returns.
Usage: python3 02_analysis.py   (reads ../data, writes ../results)
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import (simulate, metrics, paired_test, calendar_mask,
                    stress_indicator, stationary_indices, TD)

DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
RES = os.environ.get("RES_DIR", os.path.join(HERE, "..", "results"))
os.makedirs(RES, exist_ok=True)
BP = 1e-4
GAMMA = 5.0
WGRID = np.round(np.arange(0.00, 0.301, 0.01), 2)   # total band width W
DGRID = np.round(np.arange(0.01, 0.151, 0.01), 2)   # target-rule threshold delta
B_PAIRED = int(os.environ.get("B_PAIRED", 10000))
B_RESIM = int(os.environ.get("B_RESIM", 1000))
END = os.environ.get("END_DATE", "2026-09-30")


# ------------------------------------------------------------------ data
def load_px(name):
    f = os.path.join(DATA, name.replace("^", "IDX_").replace(".", "_").replace("=", "_") + ".csv")
    s = pd.read_csv(f, parse_dates=["date"]).set_index("date")["adjclose"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s[s > 0]


def load_rf_us(index):
    f = os.path.join(DATA, "FRED_DTB3.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        d.columns = ["date", "y"]
        d["date"] = pd.to_datetime(d["date"])
        y = pd.to_numeric(d["y"], errors="coerce").values / 100
        y = pd.Series(y, index=d["date"])
    else:                                            # fallback: 13-week T-bill index
        y = load_px("^IRX") / 100
    y = y.reindex(index.union(y.index)).ffill().reindex(index)
    return ((1 + y) ** (1 / TD) - 1).fillna(0).values


def pair(s, b, start=None):
    ps, pb = load_px(s), load_px(b)
    df = pd.concat([ps, pb], axis=1, join="inner").dropna()
    df.columns = ["S", "B"]
    df = df[df.index <= END]
    if start:
        df = df[df.index >= start]
    R = (df / df.shift(1)).iloc[1:]
    return R.index, R["S"].values, R["B"].values


# ------------------------------------------------------------------ strategy book
def book(dates, wstar, cup, cdn, monthly=False):
    """Returns list of (label, family, param, monitor, L, U, mode)."""
    T = len(dates)
    daily = calendar_mask(dates, "M") if monthly else np.ones(T, bool)
    lab = []
    lab.append(("Buy-and-hold", "BH", np.nan, np.zeros(T, bool), 0.0, 1.0, 0))
    for f, nm in [("M", "Monthly"), ("Q", "Quarterly"), ("A", "Annual")]:
        lab.append((nm + " rebalancing", "CAL" + f, np.nan, calendar_mask(dates, f),
                    wstar, wstar, 1))
    for d in DGRID:
        lab.append(("Threshold-to-target", "TGT", d, daily, wstar - d, wstar + d, 1))
    sh = cup / (cup + cdn)                     # lower-side share of width (static costs)
    for W in WGRID:
        lab.append(("Symmetric band", "SYM", W, daily, wstar - W / 2, wstar + W / 2, 0))
        lab.append(("Asymmetric band", "ASY", W, daily, wstar - sh * W,
                    wstar + (1 - sh) * W, 0))
    return lab


def run_book(RS, RB, lab, wstar, cup_t, cdn_t):
    K = len(lab)
    mon = np.stack([x[3] for x in lab])
    L = np.array([x[4] for x in lab], float)
    U = np.array([x[5] for x in lab], float)
    mode = np.array([x[6] for x in lab])
    return simulate(RS, RB, mon, L, U, mode, np.full(K, wstar), cup_t, cdn_t)


def ce_prefix(r, rf, upto):
    x = r[:, :upto] - rf[:upto]
    return x.mean(1) * TD - 0.5 * GAMMA * x.var(1, ddof=1) * TD


def walk_forward(dates, RS, RB, rf, lab, sim, wstar, cup_t, cdn_t, burn_years):
    """
    Out-of-sample strategies for families TGT, SYM, ASY: in each calendar
    year after the burn-in, use the parameter that maximised CE on all prior
    data; bounds are piecewise constant by year and the portfolio state is
    carried over.  Returns {family: (sim_dict, chosen params by year)} with
    simulation restricted to the OOS period.
    """
    years = pd.DatetimeIndex(dates).year.values
    first = years[0] + burn_years
    oos = years >= first
    T = len(dates)
    fam_idx = {}
    for i, x in enumerate(lab):
        fam_idx.setdefault(x[1], []).append(i)
    out = {}
    for fam in ["TGT", "SYM", "ASY"]:
        idx = fam_idx[fam]
        Lt = np.empty(T); Ut = np.empty(T); chosen = {}
        mode = lab[idx[0]][6]
        mon = lab[idx[0]][3]
        for y in np.unique(years):
            m = years == y
            if y < first:
                j = idx[0]
            else:
                upto = int(np.argmax(years >= y))
                ce = ce_prefix(sim["r"][idx], rf, upto)
                j = idx[int(np.argmax(ce))]
                chosen[int(y)] = float(lab[j][2])
            Lt[m] = lab[j][4]; Ut[m] = lab[j][5]
        s = simulate(RS, RB, mon[None, :], Lt[None, :], Ut[None, :], np.array([mode]),
                     np.array([wstar]), cup_t, cdn_t)
        out[fam] = (s, chosen)
    return out, oos


def sub(s, mask):
    return {k: v[..., mask] for k, v in s.items()}


# ------------------------------------------------------------------ one experiment
def experiment(tag, s_tk, b_tk, wstar, cost, burn, rf_kind="us", monthly=False,
               stress=None, start=None, save_paths=False):
    """cost: dict with one-way bp: Sbuy, Ssell, Bbuy, Bsell; optional 'stress_add'
    added to Ssell on real-time stress days (then a state-aware ASY is added)."""
    dates, RS, RB = pair(s_tk, b_tk, start)
    T = len(dates)
    rf = load_rf_us(dates) if rf_kind == "us" else np.zeros(T)
    ssell = np.full(T, cost["Ssell"] * BP)
    flag = None
    if stress:
        flag = stress_indicator(RS)
        ssell = ssell + flag * stress * BP
    cup_t = np.full(T, (cost["Sbuy"] + cost["Bsell"]) * BP)
    cdn_t = ssell + cost["Bbuy"] * BP
    cup0, cdn0 = cup_t[0], (cost["Ssell"] + cost["Bbuy"]) * BP
    lab = book(dates, wstar, cup0, cdn0, monthly)
    if stress:                                   # state-aware asymmetric band (same total width W)
        daily = np.ones(T, bool)
        for W in WGRID:
            sh_t = cup_t / (cup_t + cdn_t)
            lab.append(("State-aware asymmetric band", "ASYS", W, daily,
                        wstar - sh_t * W, wstar + (1 - sh_t) * W, 0))
    sim = _run_lab(RS, RB, lab, wstar, cup_t, cdn_t)
    r_tgt = wstar * RS + (1 - wstar) * RB - 1           # frictionless daily constant mix
    wf, oos = walk_forward(dates, RS, RB, rf, lab, sim, wstar, cup_t, cdn_t, burn)
    if stress:
        wf2, _ = _walk_forward_family(dates, rf, lab, sim, wstar, RS, RB, cup_t, cdn_t, burn, "ASYS")
        wf.update(wf2)
    # ---- OOS metrics table
    rows = []
    fixed = {"BH": "Buy-and-hold", "CALM": "Monthly rebalancing",
             "CALQ": "Quarterly rebalancing", "CALA": "Annual rebalancing"}
    series = {}
    for fam, name in fixed.items():
        j = [i for i, x in enumerate(lab) if x[1] == fam][0]
        s1 = sub({k: v[j:j + 1] for k, v in sim.items()}, oos)
        series[fam] = s1["r"][0]
        rows.append((name, fam, "", s1))
    j5 = [i for i, x in enumerate(lab) if x[1] == "TGT" and abs(x[2] - 0.05) < 1e-9][0]
    s1 = sub({k: v[j5:j5 + 1] for k, v in sim.items()}, oos)
    series["TGT5"] = s1["r"][0]
    rows.append(("Threshold-to-target, 5% (fixed)", "TGT5", "0.05", s1))
    for fam, name in [("TGT", "Threshold-to-target (OOS)"), ("SYM", "Symmetric band (OOS)"),
                      ("ASY", "Asymmetric band (OOS)"), ("ASYS", "State-aware asymmetric band (OOS)")]:
        if fam not in wf:
            continue
        s, ch = wf[fam]
        s1 = sub(s, oos)
        series[fam] = s1["r"][0]
        med = np.median(list(ch.values())) if ch else np.nan
        rows.append((name, fam, "%.2f" % med, s1))
    rfo, rto = rf[oos], r_tgt[oos]
    tab = []
    for name, fam, par, s1 in rows:
        m = metrics(s1, rfo, rto, GAMMA)
        tab.append(dict(strategy=name, code=fam, param_median=par,
                        **{k: float(v[0]) for k, v in m.items()}))
    tab = pd.DataFrame(tab)
    # ---- paired tests, ASY vs others (OOS)
    tests = []
    comps = [("ASY", "SYM"), ("ASY", "TGT"), ("ASY", "TGT5"), ("ASY", "CALA"),
             ("ASY", "CALQ"), ("ASY", "CALM"), ("ASY", "BH")]
    if stress:
        comps = [("ASYS", "ASY"), ("ASYS", "SYM")] + comps
    for a, b in comps:
        pt = paired_test(series[a], series[b], rfo, B=B_PAIRED, mean_block=21, gamma=GAMMA)
        tests.append(dict(a=a, b=b, dmean=pt["est"][0], dmean_lo=pt["lo"][0], dmean_hi=pt["hi"][0],
                          p_mean=pt["p"][0], dsharpe=pt["est"][1], dsharpe_lo=pt["lo"][1],
                          dsharpe_hi=pt["hi"][1], p_sharpe=pt["p"][1], dce=pt["est"][2],
                          dce_lo=pt["lo"][2], dce_hi=pt["hi"][2], p_ce=pt["p"][2]))
    tests = pd.DataFrame(tests)
    # ---- full-sample (in-sample) frontier by W for SYM / ASY
    full = metrics(sim, rf, r_tgt, GAMMA)
    fr = []
    for i, x in enumerate(lab):
        if x[1] in ("SYM", "ASY", "ASYS", "TGT"):
            fr.append(dict(family=x[1], param=x[2], ce=full["ce"][i], sharpe=full["sharpe"][i],
                           costdrag=full["costdrag"][i], turnover=full["turnover"][i],
                           te=full["te"][i], avgw=full["avgw"][i], mean=full["mean"][i]))
    fr = pd.DataFrame(fr)
    meta = dict(tag=tag, S=s_tk, B=b_tk, start=str(dates[0].date()), end=str(dates[-1].date()),
                oos_start=str(dates[oos][0].date()), T=int(T), T_oos=int(oos.sum()), wstar=wstar,
                cost=cost, c_up_bp=float(cup0 / BP), c_dn_bp=float(cdn0 / BP),
                lower_share=float(cup0 / (cup0 + cdn0)), monthly=monthly,
                stress_add=stress, stress_share=float(flag.mean()) if flag is not None else None,
                chosen={f: wf[f][1] for f in wf})
    for nm, df in [("oos", tab), ("tests", tests), ("frontier", fr)]:
        df.to_csv(os.path.join(RES, "%s_%s.csv" % (tag, nm)), index=False)
    with open(os.path.join(RES, "%s_meta.json" % tag), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return tab, tests, fr, meta


def _run_lab(RS, RB, lab, wstar, cup_t, cdn_t):
    K = len(lab); T = len(RS)
    mon = np.stack([x[3] for x in lab])
    L = np.stack([np.broadcast_to(x[4], (T,)) for x in lab]).astype(float)
    U = np.stack([np.broadcast_to(x[5], (T,)) for x in lab]).astype(float)
    mode = np.array([x[6] for x in lab])
    return simulate(RS, RB, mon, L, U, mode, np.full(K, wstar), cup_t, cdn_t)


def _walk_forward_family(dates, rf, lab, sim, wstar, RS, RB, cup_t, cdn_t, burn, fam):
    years = pd.DatetimeIndex(dates).year.values
    first = years[0] + burn
    T = len(dates)
    idx = [i for i, x in enumerate(lab) if x[1] == fam]
    Lt = np.empty(T); Ut = np.empty(T); chosen = {}
    for y in np.unique(years):
        m = years == y
        if y < first:
            j = idx[0]
        else:
            upto = int(np.argmax(years >= y))
            j = idx[int(np.argmax(ce_prefix(sim["r"][idx], rf, upto)))]
            chosen[int(y)] = float(lab[j][2])
        Lt[m] = np.broadcast_to(lab[j][4], (T,))[m]
        Ut[m] = np.broadcast_to(lab[j][5], (T,))[m]
    s = simulate(RS, RB, lab[idx[0]][3][None, :], Lt[None, :], Ut[None, :], np.array([0]),
                 np.array([wstar]), cup_t, cdn_t)
    return {fam: (s, chosen)}, None


# ------------------------------------------------------------------ mechanism: re-simulated histories
def resim(tag, s_tk, b_tk, wstar, psis, B, mean_block=63, seed=7, W=None):
    """
    Re-simulate SYM and ASY over B stationary-bootstrap histories of the joint
    (S,B) daily returns.  For each psi, both families use the same total width
    W (default: SYM's full-sample CE-optimal width under that psi on the
    actual history).  Records per history: CE(ASY) - CE(SYM), cost-drag
    difference, and the realised annualised equity-minus-bond return.
    """
    dates, RS, RB = pair(s_tk, b_tk)
    rf = load_rf_us(dates)
    T = len(RS)
    rng = np.random.default_rng(seed)
    idx = stationary_indices(T, B, mean_block, rng)
    RSb, RBb, rfb = RS[idx], RB[idx], rf[idx]
    spread = (np.log(RSb).mean(1) - np.log(RBb).mean(1)) * TD
    out = []
    for psi in psis:
        cup = 10 * BP; cdn = (10 + psi) * BP
        sh = cup / (cup + cdn)
        if W is None:
            fr = pd.read_csv(os.path.join(RES, "%s_psi%d_frontier.csv" % (tag, psi))) \
                if os.path.exists(os.path.join(RES, "%s_psi%d_frontier.csv" % (tag, psi))) else None
            Wp = float(fr[fr.family == "SYM"].sort_values("ce").iloc[-1].param) if fr is not None else 0.10
        else:
            Wp = W
        Wp = max(Wp, 0.02)
        K = 2 * B
        mon = np.ones((K, T), bool)
        L = np.r_[np.full(B, wstar - Wp / 2), np.full(B, wstar - sh * Wp)]
        U = np.r_[np.full(B, wstar + Wp / 2), np.full(B, wstar + (1 - sh) * Wp)]
        s = simulate(np.r_[RSb, RSb], np.r_[RBb, RBb], mon, L, U, np.zeros(K, int),
                     np.full(K, wstar), np.full(T, cup), np.full(T, cdn), keep="min")
        ex = s["r"] - np.r_[rfb, rfb]
        ce = ex.mean(1) * TD - 0.5 * GAMMA * ex.var(1, ddof=1) * TD
        cd = s["cost"].sum(1) / (T / TD)
        sr = ex.mean(1) / ex.std(1, ddof=1) * np.sqrt(TD)
        for b in range(B):
            out.append(dict(psi=psi, W=Wp, path=b, dce=ce[B + b] - ce[b],
                            dsharpe=sr[B + b] - sr[b], dcost=cd[B + b] - cd[b],
                            spread=spread[b]))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(RES, "%s_resim.csv" % tag), index=False)
    return df


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    PSIS = [0, 10, 25, 50, 100, 200]
    US = dict(Sbuy=5, Bsell=5, Bbuy=5)
    TW = dict(Sbuy=14.25, Ssell=24.25, Bbuy=14.25, Bsell=14.25)
    jobs = []
    for psi in PSIS:
        jobs.append(("US_psi%d" % psi, "SPY", "IEF", 0.60, dict(US, Ssell=5 + psi), 5, "us", False, None))
    jobs += [
        ("TW_statutory", "0050.TW", "00679B.TWO", 0.60, TW, 3, "tw", False, None),
        ("US_AGG_psi50", "SPY", "AGG", 0.60, dict(US, Ssell=55), 5, "us", False, None),
        ("US_TLT_psi50", "SPY", "TLT", 0.60, dict(US, Ssell=55), 5, "us", False, None),
        ("US_LONG_psi50", "VFINX", "VUSTX", 0.60, dict(US, Ssell=55), 5, "us", False, None),
        ("US_w40_psi50", "SPY", "IEF", 0.40, dict(US, Ssell=55), 5, "us", False, None),
        ("US_w80_psi50", "SPY", "IEF", 0.80, dict(US, Ssell=55), 5, "us", False, None),
        ("US_monthly_psi50", "SPY", "IEF", 0.60, dict(US, Ssell=55), 5, "us", True, None),
        ("US_stress", "SPY", "IEF", 0.60, dict(US, Ssell=5 + 25), 5, "us", False, 75),
    ]
    only = sys.argv[1:]
    for tag, s, b, w, c, burn, rk, mo, st in jobs:
        if only and tag not in only:
            continue
        try:
            tab, tests, fr, meta = experiment(tag, s, b, w, c, burn, rk, mo, st)
        except FileNotFoundError as e:
            print("SKIP", tag, "(missing data:", e, ")")
            continue
        print("\n==== %s  %s/%s  %s..%s  OOS from %s  c_up=%.1f c_dn=%.1f bp" % (
            tag, s, b, meta["start"], meta["end"], meta["oos_start"], meta["c_up_bp"], meta["c_dn_bp"]))
        print(tab[["strategy", "param_median", "mean", "vol", "sharpe", "ce", "costdrag",
                   "turnover", "te", "mdd", "avgw"]].round(4).to_string(index=False))
        print(tests[["a", "b", "dce", "dce_lo", "dce_hi", "p_ce", "dsharpe", "p_sharpe"]].round(5).to_string(index=False))
    if not only or "RESIM" in only:
        df = resim("US", "SPY", "IEF", 0.60, PSIS, B_RESIM)
        g = df.groupby("psi").agg(W=("W", "first"), dce_mean=("dce", "mean"),
                                  win=("dce", lambda x: (x > 0).mean()),
                                  dcost=("dcost", "mean"))
        print("\n==== Re-simulated histories\n", g.round(5))


# ================================================================== location vs. tilt
def decomposition(tag, s_tk, b_tk, wstar, cost, W=0.10, rf_kind="us", start=None,
                  B=None, seed=3):
    r"""
    Exact daily identity for two strategies A (asymmetric band) and S
    (symmetric band) of equal width W:
        r_A,t - r_S,t = (w_A,t-1 - w_S,t-1)(r_S,t - r_B,t)  -  (g_A,t c_A,t - g_S,t c_S,t)
                        \_____ allocation (tilt) ______/       \____ cost component ____/
    where w_.,t-1 is the post-trade weight carried into day t and g the gross
    return.  Reports annualised means with paired stationary-bootstrap CIs,
    plus the benchmarks needed for the width comparison (daily rebalancing,
    monthly rebalancing).
    """
    B = B or B_PAIRED
    dates, RS, RB = pair(s_tk, b_tk, start)
    T = len(dates)
    rf = load_rf_us(dates) if rf_kind == "us" else np.zeros(T)
    cup = (cost["Sbuy"] + cost["Bsell"]) * BP
    cdn = (cost["Ssell"] + cost["Bbuy"]) * BP
    sh = cup / (cup + cdn)
    ones = np.ones(T, bool)
    mon = np.stack([ones, ones, ones, calendar_mask(dates, "M")])
    L = np.array([wstar - W / 2, wstar - sh * W, wstar, wstar])
    U = np.array([wstar + W / 2, wstar + (1 - sh) * W, wstar, wstar])
    s = simulate(RS, RB, mon, L, U, np.array([0, 0, 1, 1]), np.full(4, wstar),
                 np.full(T, cup), np.full(T, cdn))
    wprev = np.concatenate([np.full((4, 1), wstar), s["w"][:, :-1]], axis=1)
    g = wprev * RS + (1 - wprev) * RB
    alloc = (wprev[1] - wprev[0]) * (RS - RB)
    costc = -(g[1] * s["cost"][1] - g[0] * s["cost"][0])
    dr = s["r"][1] - s["r"][0]
    assert np.allclose(dr, alloc + costc, atol=1e-12)
    rng = np.random.default_rng(seed)
    idx = stationary_indices(T, B, 21, rng)
    out = {}
    for nm, x in [("dmean", dr), ("alloc", alloc), ("cost", costc)]:
        bs = x[idx].mean(1) * TD
        out[nm] = (x.mean() * TD, *np.percentile(bs, [2.5, 97.5]))
    m = metrics(s, rf, wstar * RS + (1 - wstar) * RB - 1, GAMMA)
    ex = s["r"] - rf
    bsce = []
    for i in (0, 1):
        e = ex[i][idx]
        bsce.append(e.mean(1) * TD - 0.5 * GAMMA * e.var(1, ddof=1) * TD)
    dce_bs = bsce[1] - bsce[0]
    dce = m["ce"][1] - m["ce"][0]
    res = dict(tag=tag, W=W, c_up_bp=cup / BP, c_dn_bp=cdn / BP, lower_share=sh,
               L_asy=wstar - sh * W, U_asy=wstar + (1 - sh) * W,
               cost_daily_bp=m["costdrag"][2] / BP, cost_monthly_bp=m["costdrag"][3] / BP,
               cost_sym_bp=m["costdrag"][0] / BP, cost_asy_bp=m["costdrag"][1] / BP,
               turn_daily=m["turnover"][2], turn_monthly=m["turnover"][3],
               turn_sym=m["turnover"][0], turn_asy=m["turnover"][1],
               te_sym=m["te"][0], te_asy=m["te"][1], te_monthly=m["te"][3],
               avgw_sym=m["avgw"][0], avgw_asy=m["avgw"][1],
               frac_upper_sym=float((s["w"][0] > wstar).mean()),
               dce=dce, dce_lo=np.percentile(dce_bs, 2.5), dce_hi=np.percentile(dce_bs, 97.5),
               p_dce=float((np.abs(dce_bs - dce_bs.mean()) >= abs(dce)).mean()),
               sharpe_sym=m["sharpe"][0], sharpe_asy=m["sharpe"][1],
               ce_sym=m["ce"][0], ce_asy=m["ce"][1],
               start=str(dates[0].date()), end=str(dates[-1].date()), T=T)
    for nm in out:
        res[nm], res[nm + "_lo"], res[nm + "_hi"] = out[nm]
    return res


def resim_fixed(s_tk, b_tk, wstar, psis, B, W=0.10, mean_block=63, seed=7):
    """Re-simulated histories at fixed width W; per-history differences
    ASY - SYM in CE, mean, cost drag, average weight, and realised spread."""
    dates, RS, RB = pair(s_tk, b_tk)
    rf = load_rf_us(dates)
    T = len(RS)
    rng = np.random.default_rng(seed)
    idx = stationary_indices(T, B, mean_block, rng)
    RSb, RBb, rfb = RS[idx], RB[idx], rf[idx]
    spread = (RSb.mean(1) - RBb.mean(1)) * TD
    rows = []
    for psi in psis:
        cup = 10 * BP; cdn = (10 + psi) * BP; sh = cup / (cup + cdn)
        ce = np.empty(2 * B); mu = np.empty(2 * B); cd = np.empty(2 * B); aw = np.empty(2 * B)
        for lo in range(0, B, 200):                      # chunks keep memory bounded
            hi = min(B, lo + 200); n = hi - lo; K = 2 * n
            L = np.r_[np.full(n, wstar - W / 2), np.full(n, wstar - sh * W)]
            U = np.r_[np.full(n, wstar + W / 2), np.full(n, wstar + (1 - sh) * W)]
            s = simulate(np.r_[RSb[lo:hi], RSb[lo:hi]], np.r_[RBb[lo:hi], RBb[lo:hi]],
                         np.ones((K, T), bool), L, U, np.zeros(K, int), np.full(K, wstar),
                         np.full(T, cup), np.full(T, cdn), keep="all")
            ex = s["r"] - np.r_[rfb[lo:hi], rfb[lo:hi]]
            sel = np.r_[np.arange(lo, hi), B + np.arange(lo, hi)]
            ce[sel] = ex.mean(1) * TD - 0.5 * GAMMA * ex.var(1, ddof=1) * TD
            mu[sel] = s["r"].mean(1) * TD
            cd[sel] = s["cost"].sum(1) / (T / TD)
            aw[sel] = s["w"].mean(1)
            del s, ex
        for b in range(B):
            rows.append(dict(psi=psi, W=W, path=b, dce=ce[B + b] - ce[b], dmean=mu[B + b] - mu[b],
                             dcost=cd[B + b] - cd[b], davgw=aw[B + b] - aw[b], spread=spread[b]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "US_resim_W%02d.csv" % int(round(W * 100))), index=False)
    return df


def run_location():
    US = dict(Sbuy=5, Bsell=5, Bbuy=5)
    TW = dict(Sbuy=14.25, Ssell=24.25, Bbuy=14.25, Bsell=14.25)
    specs = [("US, psi=0", "SPY", "IEF", 0.6, dict(US, Ssell=5), "us"),
             ("US, psi=50", "SPY", "IEF", 0.6, dict(US, Ssell=55), "us"),
             ("US, psi=200", "SPY", "IEF", 0.6, dict(US, Ssell=205), "us"),
             ("Taiwan, statutory", "0050.TW", "00679B.TWO", 0.6, TW, "tw"),
             ("US, AGG, psi=50", "SPY", "AGG", 0.6, dict(US, Ssell=55), "us"),
             ("US, TLT, psi=50", "SPY", "TLT", 0.6, dict(US, Ssell=55), "us"),
             ("US long, psi=50", "VFINX", "VUSTX", 0.6, dict(US, Ssell=55), "us"),
             ("US, w*=0.4, psi=50", "SPY", "IEF", 0.4, dict(US, Ssell=55), "us"),
             ("US, w*=0.8, psi=50", "SPY", "IEF", 0.8, dict(US, Ssell=55), "us")]
    rows = []
    for W in (0.10, 0.06, 0.20):
        for tag, s, b, w, c, rk in specs:
            if W != 0.10 and tag not in ("US, psi=50", "Taiwan, statutory"):
                continue
            rows.append(decomposition(tag, s, b, w, c, W=W, rf_kind=rk))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "location_decomposition.csv"), index=False)
    return df


if __name__ == "__main__" and os.environ.get("RUN_LOCATION", "1") == "1":
    d = run_location()
    pd.set_option("display.width", 250)
    print("\n==== Location vs tilt decomposition (annualised, bp unless noted)")
    v = d.copy()
    for c in ["dmean", "dmean_lo", "dmean_hi", "alloc", "alloc_lo", "alloc_hi", "cost", "cost_lo",
              "cost_hi", "dce", "dce_lo", "dce_hi"]:
        v[c] = v[c] * 1e4
    print(v[["tag", "W", "lower_share", "cost_monthly_bp", "cost_daily_bp", "cost_sym_bp", "cost_asy_bp",
             "avgw_sym", "avgw_asy", "frac_upper_sym", "dmean", "alloc", "cost", "cost_lo", "cost_hi",
             "dce", "dce_lo", "dce_hi", "p_dce"]].round(3).to_string(index=False))
    r = resim_fixed("SPY", "IEF", 0.60, [0, 10, 25, 50, 100, 200], B_RESIM, W=0.10)
    print("\n==== Re-simulated histories at W = 0.10")
    for psi, g in r.groupby("psi"):
        if psi == 0:
            continue
        X = np.c_[np.ones(len(g)), g.spread.values]
        beta = np.linalg.lstsq(X, g.dce.values, rcond=None)[0]
        corr_w = np.corrcoef(g.dmean, g.davgw * g.spread)[0, 1]
        print("psi=%3d  dCE mean %+.2f bp  P(dCE>0)=%.3f  dCost mean %+.3f bp  dAvgW %+.2f pp  "
              "slope dCE on spread %+.4f  P(dCE>0|spread>median)=%.3f  P(dCE>0|spread<=median)=%.3f" % (
                  psi, g.dce.mean() * 1e4, (g.dce > 0).mean(), g.dcost.mean() * 1e4, g.davgw.mean() * 100,
                  beta[1], (g.dce[g.spread > g.spread.median()] > 0).mean(),
                  (g.dce[g.spread <= g.spread.median()] > 0).mean()))
