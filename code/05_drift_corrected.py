#!/usr/bin/env python3
r"""
05_drift_corrected.py -- the drift-corrected band (DCB).

Rule.  For total width W, the band is [omega - W/2 - d_y, omega + W/2 - d_y],
trading to the nearest edge, where the offset d_y used throughout calendar
year y is the average over-weight, mean_t (w_t - omega), that a centred band
of the same width (SYM) produced on ALL days before year y (d = 0 in the first
year).  The rule uses only past information.

Evaluation.  Implementation loss relative to the frictionless target,
    IL = annual cost drag + (gamma/2) * TE^2,
the empirical counterpart of objective (1); TE is the annualised standard
deviation of the return difference from the daily-rebalanced target
portfolio.  Unlike CE, IL does not reward a lucky equity tilt.

Outputs (../results): dcb_full.csv, dcb_frontier.csv, dcb_oos.csv,
dcb_oos_tests.csv, dcb_resim.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("an", os.path.join(HERE, "02_analysis.py"))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
from engine import simulate, metrics, stationary_indices, calendar_mask, TD

RES, BP, GAMMA = an.RES, an.BP, an.GAMMA
WGRID = np.round(np.arange(0.02, 0.301, 0.01), 2)
B_PAIRED = an.B_PAIRED
B_RESIM = an.B_RESIM


def year_index(dates):
    y = pd.DatetimeIndex(dates).year.values
    return y


def dcb_bounds(wsym, years, omega, W):
    """Real-time offsets from the SYM weight path (same width)."""
    T = len(wsym)
    dev = wsym - omega
    L = np.empty(T); U = np.empty(T); d_by_year = {}
    cs = np.concatenate([[0.0], np.cumsum(dev)])
    for y in np.unique(years):
        m = years == y
        t0 = int(np.argmax(m))
        d = cs[t0] / t0 if t0 > 0 else 0.0
        d_by_year[int(y)] = d
        L[m] = omega - W / 2 - d; U[m] = omega + W / 2 - d
    return L, U, d_by_year


def run_family(RS, RB, years, omega, cup, cdn, Ws):
    """Simulate SYM, ASY, DCB for widths Ws on one history."""
    T = len(RS); K = len(Ws)
    sh = cup / (cup + cdn)
    ones = np.ones((K, T), bool)
    sym = simulate(RS, RB, ones, omega - Ws / 2, omega + Ws / 2, np.zeros(K, int),
                   np.full(K, omega), np.full(T, cup), np.full(T, cdn))
    asy = simulate(RS, RB, ones, omega - sh * Ws, omega + (1 - sh) * Ws, np.zeros(K, int),
                   np.full(K, omega), np.full(T, cup), np.full(T, cdn))
    Lb = np.empty((K, T)); Ub = np.empty((K, T)); offs = []
    for k, W in enumerate(Ws):
        Lb[k], Ub[k], d = dcb_bounds(sym["w"][k], years, omega, W); offs.append(d)
    dcb = simulate(RS, RB, ones, Lb, Ub, np.zeros(K, int), np.full(K, omega),
                   np.full(T, cup), np.full(T, cdn))
    return sym, asy, dcb, (Lb, Ub, offs)


def il_of(m):
    return m["costdrag"] + 0.5 * GAMMA * m["te"] ** 2


def boot_diff(ra, rb, rt, ca, cb, idx):
    """Bootstrap CIs for DCB - SYM differences in TE, cost drag and IL."""
    aa, ab = (ra - rt)[idx], (rb - rt)[idx]
    tea = aa.std(1, ddof=1) * np.sqrt(TD); teb = ab.std(1, ddof=1) * np.sqrt(TD)
    cda = ca[idx].mean(1) * TD; cdb = cb[idx].mean(1) * TD
    ila = cda + 0.5 * GAMMA * tea ** 2; ilb = cdb + 0.5 * GAMMA * teb ** 2
    # point estimates on the original sample
    T = ra.shape[-1]
    te_a = (ra - rt).std(ddof=1) * np.sqrt(TD); te_b = (rb - rt).std(ddof=1) * np.sqrt(TD)
    est = {"dte": te_a - te_b, "dcost": (ca.mean() - cb.mean()) * TD,
           "dil": (ca.mean() * TD + 0.5 * GAMMA * te_a ** 2) - (cb.mean() * TD + 0.5 * GAMMA * te_b ** 2)}
    out = {}
    for nm, x in [("dte", tea - teb), ("dcost", cda - cdb), ("dil", ila - ilb)]:
        out[nm] = np.percentile(x, [2.5, 97.5])
        out[nm + "_p"] = float((np.abs(x - x.mean()) >= abs(est[nm])).mean())   # null-centred, two-sided
    return out


SPECS = [("US, psi=50", "SPY", "IEF", 0.6, dict(Sbuy=5, Bsell=5, Bbuy=5, Ssell=55), "us", 5),
         ("US, psi=200", "SPY", "IEF", 0.6, dict(Sbuy=5, Bsell=5, Bbuy=5, Ssell=205), "us", 5),
         ("Taiwan, statutory", "0050.TW", "00679B.TWO", 0.6,
          dict(Sbuy=14.25, Ssell=24.25, Bbuy=14.25, Bsell=14.25), "tw", 3),
         ("US, AGG, psi=50", "SPY", "AGG", 0.6, dict(Sbuy=5, Bsell=5, Bbuy=5, Ssell=55), "us", 5),
         ("US, TLT, psi=50", "SPY", "TLT", 0.6, dict(Sbuy=5, Bsell=5, Bbuy=5, Ssell=55), "us", 5),
         ("US long, psi=50", "VFINX", "VUSTX", 0.6, dict(Sbuy=5, Bsell=5, Bbuy=5, Ssell=55), "us", 5),
         ("US, target 0.4, psi=50", "SPY", "IEF", 0.4, dict(Sbuy=5, Bsell=5, Bbuy=5, Ssell=55), "us", 5),
         ("US, target 0.8, psi=50", "SPY", "IEF", 0.8, dict(Sbuy=5, Bsell=5, Bbuy=5, Ssell=55), "us", 5)]


def full_and_oos():
    full_rows, fr_rows, oos_rows, test_rows = [], [], [], []
    for tag, s, b, omega, c, rk, burn in SPECS:
        dates, RS, RB = an.pair(s, b)
        T = len(dates); years = year_index(dates)
        rf = an.load_rf_us(dates) if rk == "us" else np.zeros(T)
        cup = (c["Sbuy"] + c["Bsell"]) * BP; cdn = (c["Ssell"] + c["Bbuy"]) * BP
        rt = omega * RS + (1 - omega) * RB - 1
        sym, asy, dcb, (Lb, Ub, offs) = run_family(RS, RB, years, omega, cup, cdn, WGRID)
        mon = np.stack([calendar_mask(dates, "M"), calendar_mask(dates, "A"), np.ones(T, bool)])
        cal = simulate(RS, RB, mon, np.full(3, omega), np.full(3, omega), np.ones(3, int),
                       np.full(3, omega), np.full(T, cup), np.full(T, cdn))
        M = {k: metrics(v, rf, rt, GAMMA) for k, v in [("SYM", sym), ("ASY", asy), ("DCB", dcb), ("CAL", cal)]}
        for fam in ("SYM", "ASY", "DCB"):
            for k, W in enumerate(WGRID):
                fr_rows.append(dict(tag=tag, family=fam, W=W, cost=M[fam]["costdrag"][k], te=M[fam]["te"][k],
                                    il=il_of(M[fam])[k], avgw=M[fam]["avgw"][k], ce=M[fam]["ce"][k],
                                    turnover=M[fam]["turnover"][k]))
        # full-sample comparison at W = 10%
        k = int(np.argmin(np.abs(WGRID - 0.10)))
        rng = np.random.default_rng(11)
        idx = stationary_indices(T, B_PAIRED, 21, rng)
        bd = boot_diff(dcb["r"][k], sym["r"][k], rt, dcb["cost"][k], sym["cost"][k], idx)
        row = dict(tag=tag, T=T, start=str(dates[0].date()), end=str(dates[-1].date()), omega=omega,
                   final_offset=list(offs[k].values())[-1])
        for fam in ("SYM", "ASY", "DCB"):
            row.update({fam + "_cost": M[fam]["costdrag"][k], fam + "_te": M[fam]["te"][k],
                        fam + "_il": il_of(M[fam])[k], fam + "_avgw": M[fam]["avgw"][k],
                        fam + "_ce": M[fam]["ce"][k], fam + "_turn": M[fam]["turnover"][k]})
        for j, nm in enumerate(["MON", "ANN", "DAY"]):
            row.update({nm + "_cost": M["CAL"]["costdrag"][j], nm + "_te": M["CAL"]["te"][j],
                        nm + "_il": il_of(M["CAL"])[j]})
        for key, v in bd.items():
            if key.endswith("_p"):
                row[key] = v
            else:
                row[key + "_lo"], row[key + "_hi"] = v
        full_rows.append(row)
        # ---- out-of-sample: choose W (or delta) each January by minimising IL on prior data
        first = years[0] + burn; oos = years >= first
        rt_o = rt
        def il_prefix(sim, upto):
            a = sim["r"][:, :upto] - rt_o[:upto]
            te = a.std(1, ddof=1) * np.sqrt(TD)
            return sim["cost"][:, :upto].mean(1) * TD + 0.5 * GAMMA * te ** 2
        tg = simulate(RS, RB, np.ones((len(an.DGRID), T), bool), omega - an.DGRID, omega + an.DGRID,
                      np.ones(len(an.DGRID), int), np.full(len(an.DGRID), omega), np.full(T, cup), np.full(T, cdn))
        fams = {"SYM": (sym, omega - WGRID[:, None] / 2 + 0 * Lb, omega + WGRID[:, None] / 2 + 0 * Ub, 0, WGRID),
                "ASY": (asy, omega - (cup / (cup + cdn)) * WGRID[:, None] + 0 * Lb,
                        omega + (cdn / (cup + cdn)) * WGRID[:, None] + 0 * Ub, 0, WGRID),
                "DCB": (dcb, Lb, Ub, 0, WGRID),
                "TGT": (tg, (omega - an.DGRID)[:, None] + 0 * Lb[:len(an.DGRID)],
                        (omega + an.DGRID)[:, None] + 0 * Ub[:len(an.DGRID)], 1, an.DGRID)}
        series = {}
        chosen_all = {}
        for fam, (sim, LL, UU, mode, grid) in fams.items():
            Lt = np.empty(T); Ut = np.empty(T); ch = {}
            for y in np.unique(years):
                m = years == y
                if y < first:
                    j = int(np.argmin(np.abs(grid - 0.10)))
                else:
                    upto = int(np.argmax(years >= y)); j = int(np.argmin(il_prefix(sim, upto)))
                    ch[int(y)] = float(grid[j])
                Lt[m] = LL[j][m]; Ut[m] = UU[j][m]
            so = simulate(RS, RB, np.ones((1, T), bool), Lt[None], Ut[None], np.array([mode]),
                          np.array([omega]), np.full(T, cup), np.full(T, cdn))
            series[fam] = so; chosen_all[fam] = ch
        for j, nm in enumerate(["MON", "ANN"]):
            series[nm] = {kk: v[j:j + 1] for kk, v in cal.items()}
        series["BH"] = simulate(RS, RB, np.zeros((1, T), bool), np.array([0.0]), np.array([1.0]),
                                np.zeros(1, int), np.array([omega]), np.full(T, cup), np.full(T, cdn))
        for fam, so in series.items():
            s1 = {kk: v[:, oos] for kk, v in so.items()}
            mm = metrics(s1, rf[oos], rt[oos], GAMMA)
            oos_rows.append(dict(tag=tag, family=fam, oos_start=str(dates[oos][0].date()), end=str(dates[-1].date()),
                                 median_param=float(np.median(list(chosen_all[fam].values()))) if fam in chosen_all else np.nan,
                                 mean=mm["mean"][0], vol=mm["vol"][0], sharpe=mm["sharpe"][0], ce=mm["ce"][0],
                                 cost=mm["costdrag"][0], turnover=mm["turnover"][0], te=mm["te"][0],
                                 il=(mm["costdrag"] + 0.5 * GAMMA * mm["te"] ** 2)[0], mdd=mm["mdd"][0], avgw=mm["avgw"][0]))
        Toos = int(oos.sum())
        rng = np.random.default_rng(12)
        idx = stationary_indices(Toos, B_PAIRED, 21, rng)
        for a, bb in [("SYM", "ASY"), ("SYM", "MON"), ("SYM", "ANN"), ("SYM", "TGT"), ("SYM", "DCB")]:
            bd = boot_diff(series[a]["r"][0, oos], series[bb]["r"][0, oos], rt[oos],
                           series[a]["cost"][0, oos], series[bb]["cost"][0, oos], idx)
            ra, rb_ = series[a]["r"][0, oos], series[bb]["r"][0, oos]
            def il1(r, cst):
                return cst.mean() * TD + 0.5 * GAMMA * ((r - rt[oos]).std(ddof=1) * np.sqrt(TD)) ** 2
            test_rows.append(dict(tag=tag, a=a, b=bb, dil=il1(ra, series[a]["cost"][0, oos]) - il1(rb_, series[bb]["cost"][0, oos]),
                                  dil_lo=bd["dil"][0], dil_hi=bd["dil"][1], p_dil=bd["dil_p"],
                                  dte=(ra - rt[oos]).std(ddof=1) * np.sqrt(TD) - (rb_ - rt[oos]).std(ddof=1) * np.sqrt(TD),
                                  dte_lo=bd["dte"][0], dte_hi=bd["dte"][1]))
        print("done", tag, "chosen DCB:", chosen_all["DCB"])
    pd.DataFrame(full_rows).to_csv(os.path.join(RES, "dcb_full.csv"), index=False)
    pd.DataFrame(fr_rows).to_csv(os.path.join(RES, "dcb_frontier.csv"), index=False)
    pd.DataFrame(oos_rows).to_csv(os.path.join(RES, "dcb_oos.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(RES, "dcb_oos_tests.csv"), index=False)


def resim(W=0.10, B=None, seed=7, mean_block=63):
    """Re-simulated SPY/IEF histories; pseudo-years of 252 days for DCB updates."""
    B = B or B_RESIM
    dates, RS, RB = an.pair("SPY", "IEF")
    T = len(RS); omega = 0.6
    cup, cdn = 10 * BP, 60 * BP
    rng = np.random.default_rng(seed)
    idx = stationary_indices(T, B, mean_block, rng)
    years = np.arange(T) // 252
    rows = []
    for lo in range(0, B, 100):
        hi = min(B, lo + 100); n = hi - lo
        RSb, RBb = RS[idx[lo:hi]], RB[idx[lo:hi]]
        ones = np.ones((n, T), bool)
        sym = simulate(RSb, RBb, ones, np.full(n, omega - W / 2), np.full(n, omega + W / 2),
                       np.zeros(n, int), np.full(n, omega), np.full(T, cup), np.full(T, cdn))
        sh = cup / (cup + cdn)
        asy = simulate(RSb, RBb, ones, np.full(n, omega - sh * W), np.full(n, omega + (1 - sh) * W),
                       np.zeros(n, int), np.full(n, omega), np.full(T, cup), np.full(T, cdn))
        Lb = np.empty((n, T)); Ub = np.empty((n, T))
        for i in range(n):
            Lb[i], Ub[i], _ = dcb_bounds(sym["w"][i], years, omega, W)
        dcb = simulate(RSb, RBb, ones, Lb, Ub, np.zeros(n, int), np.full(n, omega),
                       np.full(T, cup), np.full(T, cdn))
        rt = omega * RSb + (1 - omega) * RBb - 1
        for nm, s in [("SYM", sym), ("ASY", asy), ("DCB", dcb)]:
            te = (s["r"] - rt).std(1, ddof=1) * np.sqrt(TD)
            cd = s["cost"].mean(1) * TD
            for i in range(n):
                rows.append(dict(path=lo + i, family=nm, te=te[i], cost=cd[i],
                                 il=cd[i] + 0.5 * GAMMA * te[i] ** 2, avgw=s["w"][i].mean(),
                                 spread=(RSb[i].mean() - RBb[i].mean()) * TD))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "dcb_resim.csv"), index=False)
    return df


def location_sweep():
    """IL, TE and cost as a function of the lower-side share theta at fixed width."""
    TH = np.round(np.arange(0.10, 0.901, 0.05), 2)
    rows = []
    for tag, s, b, omega, c, rk, burn in SPECS:
        dates, RS, RB = an.pair(s, b)
        T = len(RS); rt = omega * RS + (1 - omega) * RB - 1
        cup = (c["Sbuy"] + c["Bsell"]) * BP; cdn = (c["Ssell"] + c["Bbuy"]) * BP
        for W in (0.06, 0.10, 0.20):
            K = len(TH)
            sim = simulate(RS, RB, np.ones((K, T), bool), omega - TH * W, omega + (1 - TH) * W,
                           np.zeros(K, int), np.full(K, omega), np.full(T, cup), np.full(T, cdn))
            te = (sim["r"] - rt).std(1, ddof=1) * np.sqrt(TD); cd = sim["cost"].mean(1) * TD
            for k, th in enumerate(TH):
                rows.append(dict(tag=tag, W=W, theta=th, theory=cup / (cup + cdn), te=te[k], cost=cd[k],
                                 il=cd[k] + 0.5 * GAMMA * te[k] ** 2, avgw=sim["w"][k].mean()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "location_sweep.csv"), index=False)
    return df


def mechanism():
    """Volatility on days the centred 10% band is below vs at/above target."""
    rows = []
    for tag, s, b, omega, c, rk, burn in SPECS:
        dates, RS, RB = an.pair(s, b); T = len(RS)
        cup = (c["Sbuy"] + c["Bsell"]) * BP; cdn = (c["Ssell"] + c["Bbuy"]) * BP
        sim = simulate(RS, RB, np.ones((1, T), bool), np.array([omega - 0.05]), np.array([omega + 0.05]),
                       np.zeros(1, int), np.array([omega]), np.full(T, cup), np.full(T, cdn))
        w = np.r_[omega, sim["w"][0, :-1]]; x2 = (RS - RB) ** 2; lo = w < omega
        rows.append(dict(tag=tag, share_below=lo.mean(), msq_below=x2[lo].mean(), msq_above=x2[~lo].mean(),
                         ratio=x2[lo].mean() / x2[~lo].mean()))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "mechanism.csv"), index=False)
    return df


if __name__ == "__main__":
    print(mechanism().round(3).to_string())
    location_sweep()
    full_and_oos()
    d = resim()
    p = d.pivot(index="path", columns="family", values=["te", "il", "cost", "avgw"])
    print("P(IL_SYM<IL_ASY)=%.3f  P(TE_SYM<TE_ASY)=%.3f" % ((p.il.SYM < p.il.ASY).mean(), (p.te.SYM < p.te.ASY).mean()))
    print("P(TE_DCB<TE_SYM)=%.3f  P(IL_DCB<IL_SYM)=%.3f  P(IL_DCB<IL_ASY)=%.3f  mean dIL=%.3f bp  mean avgw SYM %.4f ASY %.4f DCB %.4f" % (
        (p.te.DCB < p.te.SYM).mean(), (p.il.DCB < p.il.SYM).mean(), (p.il.DCB < p.il.ASY).mean(),
        1e4 * (p.il.DCB - p.il.SYM).mean(), p.avgw.SYM.mean(), p.avgw.ASY.mean(), p.avgw.DCB.mean()))
