#!/usr/bin/env python3
"""
06_robustness.py -- robustness and mechanism checks for the implementation-loss results.

(a) gamma in {2, 5, 10} and daily vs month-end monitoring:
    IL-optimal band location (full sample) and out-of-sample ranking.
(b) Sampling distribution of the IL-optimal location: 1,000 stationary-bootstrap
    histories (mean block 63 days), W = 10%.
(c) Mechanism: drift-corrected vs centred band under iid resampling (mean block
    1 day, which destroys volatility clustering) and block resampling (63 days).
Outputs: rob_location.csv, rob_oos.csv, rob_theta_boot.csv, rob_mechanism.csv
Run after 02 and 05.
"""
import os, sys
import numpy as np
import pandas as pd
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("d5", os.path.join(HERE, "05_drift_corrected.py"))
d5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d5)
an = d5.an
from engine import simulate, stationary_indices, calendar_mask, TD

RES, BP = an.RES, an.BP
TH = np.round(np.arange(0.10, 0.901, 0.05), 2)
GAMMAS = (2.0, 5.0, 10.0)
B = an.B_RESIM


def spec_data(sp):
    tag, s, b, omega, c, rk, burn = sp
    dates, RS, RB = an.pair(s, b)
    cup = (c["Sbuy"] + c["Bsell"]) * BP; cdn = (c["Ssell"] + c["Bbuy"]) * BP
    return tag, dates, RS, RB, omega, cup, cdn, burn


def il(cost_daily, r, rt, gamma):
    te = (r - rt).std(-1, ddof=1) * np.sqrt(TD)
    return cost_daily.mean(-1) * TD + 0.5 * gamma * te ** 2


# ------------------------------------------------------------------ (a) location + OOS under gamma / monitoring
def location_and_oos():
    loc_rows, oos_rows = [], []
    for sp in d5.SPECS:
        tag, dates, RS, RB, omega, cup, cdn, burn = spec_data(sp)
        T = len(RS); rt = omega * RS + (1 - omega) * RB - 1
        years = pd.DatetimeIndex(dates).year.values
        for mon_name in ("daily", "monthly"):
            mon = np.ones(T, bool) if mon_name == "daily" else calendar_mask(dates, "M")
            K = len(TH)
            sw = simulate(RS, RB, np.tile(mon, (K, 1)), omega - TH * 0.10, omega + (1 - TH) * 0.10,
                          np.zeros(K, int), np.full(K, omega), np.full(T, cup), np.full(T, cdn), keep="min")
            sh = cup / (cup + cdn)
            asy = simulate(RS, RB, mon[None], np.array([omega - sh * 0.10]), np.array([omega + (1 - sh) * 0.10]),
                           np.zeros(1, int), np.array([omega]), np.full(T, cup), np.full(T, cdn), keep="min")
            for g in GAMMAS:
                ilv = il(sw["cost"], sw["r"], rt, g)
                loc_rows.append(dict(tag=tag, monitoring=mon_name, gamma=g, opt_share=TH[np.argmin(ilv)],
                                     il_opt=ilv.min(), il_centred=ilv[np.isclose(TH, 0.5)][0],
                                     il_theory=il(asy["cost"], asy["r"], rt, g)[0], theory_share=sh))
            # ---- OOS: families over width grid with this monitoring
            Wg = d5.WGRID; Kw = len(Wg); monK = np.tile(mon, (Kw, 1))
            sym = simulate(RS, RB, monK, omega - Wg / 2, omega + Wg / 2, np.zeros(Kw, int), np.full(Kw, omega),
                           np.full(T, cup), np.full(T, cdn))
            asyW = simulate(RS, RB, monK, omega - sh * Wg, omega + (1 - sh) * Wg, np.zeros(Kw, int),
                            np.full(Kw, omega), np.full(T, cup), np.full(T, cdn), keep="min")
            Lb = np.empty((Kw, T)); Ub = np.empty((Kw, T))
            for k, W in enumerate(Wg):
                Lb[k], Ub[k], _ = d5.dcb_bounds(sym["w"][k], years, omega, W)
            dcb = simulate(RS, RB, monK, Lb, Ub, np.zeros(Kw, int), np.full(Kw, omega),
                           np.full(T, cup), np.full(T, cdn), keep="min")
            Dg = an.DGRID; Kd = len(Dg)
            tgt = simulate(RS, RB, np.tile(mon, (Kd, 1)), omega - Dg, omega + Dg, np.ones(Kd, int),
                           np.full(Kd, omega), np.full(T, cup), np.full(T, cdn), keep="min")
            calm = calendar_mask(dates, "M"); cala = calendar_mask(dates, "A")
            cal = simulate(RS, RB, np.stack([calm, cala]), np.full(2, omega), np.full(2, omega), np.ones(2, int),
                           np.full(2, omega), np.full(T, cup), np.full(T, cdn), keep="min")
            first = years[0] + burn; oos = years >= first
            fams = {"SYM": (sym, omega - Wg[:, None] / 2 + 0 * Lb, omega + Wg[:, None] / 2 + 0 * Ub, 0),
                    "ASY": (asyW, omega - sh * Wg[:, None] + 0 * Lb, omega + (1 - sh) * Wg[:, None] + 0 * Ub, 0),
                    "DCB": (dcb, Lb, Ub, 0),
                    "TGT": (tgt, np.tile((omega - Dg)[:, None], (1, T)), np.tile((omega + Dg)[:, None], (1, T)), 1)}
            for g in GAMMAS:
                res = {}
                for f, (sim, LL, UU, mode) in fams.items():
                    Lt = np.empty(T); Ut = np.empty(T)
                    for y in np.unique(years):
                        m = years == y
                        if y < first:              # burn-in: same convention as 05 (parameter nearest 0.10)
                            j = int(np.argmin(np.abs((Dg if f == "TGT" else Wg) - 0.10)))
                        else:
                            upto = int(np.argmax(years >= y))
                            j = int(np.argmin(il(sim["cost"][:, :upto], sim["r"][:, :upto], rt[:upto], g)))
                        Lt[m] = LL[j][m]; Ut[m] = UU[j][m]
                    so = simulate(RS, RB, mon[None], Lt[None], Ut[None], np.array([mode]), np.array([omega]),
                                  np.full(T, cup), np.full(T, cdn), keep="min")
                    res[f] = il(so["cost"][0, oos], so["r"][0, oos], rt[oos], g)
                res["MON"] = il(cal["cost"][0, oos], cal["r"][0, oos], rt[oos], g)
                res["ANN"] = il(cal["cost"][1, oos], cal["r"][1, oos], rt[oos], g)
                best = min(res, key=res.get)
                rank = sorted(res, key=res.get)
                oos_rows.append(dict(tag=tag, monitoring=mon_name, gamma=g, best=best,
                                     red_vs_mon=1 - res["SYM"] / res["MON"],
                                     sym_lt_asy=res["SYM"] < res["ASY"], ranking=">".join(rank), **{"il_" + k: v for k, v in res.items()}))
        print("done", tag)
    pd.DataFrame(loc_rows).to_csv(os.path.join(RES, "rob_location.csv"), index=False)
    pd.DataFrame(oos_rows).to_csv(os.path.join(RES, "rob_oos.csv"), index=False)


# ------------------------------------------------------------------ (b) sampling distribution of the optimal location
def theta_boot(tags=("US, psi=50", "Taiwan, statutory"), W=0.10, seed=21):
    rows = []
    for sp in d5.SPECS:
        if sp[0] not in tags:
            continue
        tag, dates, RS, RB, omega, cup, cdn, burn = spec_data(sp)
        T = len(RS); sh = cup / (cup + cdn)
        idx = stationary_indices(T, B, 63, np.random.default_rng(seed))
        grid = np.r_[sh, TH]                      # theory share first, then the grid
        Kt = len(grid)
        for lo in range(0, B, 50):
            hi = min(B, lo + 50); n = hi - lo
            RSb = np.repeat(RS[idx[lo:hi]], Kt, axis=0); RBb = np.repeat(RB[idx[lo:hi]], Kt, axis=0)
            th = np.tile(grid, n)
            s = simulate(RSb, RBb, np.ones((n * Kt, T), bool), omega - th * W, omega + (1 - th) * W,
                         np.zeros(n * Kt, int), np.full(n * Kt, omega), np.full(T, cup), np.full(T, cdn), keep="min")
            rt = omega * RSb + (1 - omega) * RBb - 1
            ilv = il(s["cost"], s["r"], rt, 5.0).reshape(n, Kt)
            for i in range(n):
                g = ilv[i, 1:]
                rows.append(dict(tag=tag, path=lo + i, opt_share=TH[np.argmin(g)],
                                 il_theory=ilv[i, 0], il_centred=g[np.isclose(TH, 0.5)][0], il_opt=g.min()))
        print("theta boot done", tag)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "rob_theta_boot.csv"), index=False)
    return df


# ------------------------------------------------------------------ (c) mechanism: iid vs block resampling
def mechanism_resampling(W=0.10, seed=7):
    sp = [x for x in d5.SPECS if x[0] == "US, psi=50"][0]
    tag, dates, RS, RB, omega, cup, cdn, burn = spec_data(sp)
    T = len(RS); years = np.arange(T) // 252
    rows = []
    for blk in (1, 63):
        idx = stationary_indices(T, B, blk, np.random.default_rng(seed))
        for lo in range(0, B, 100):
            hi = min(B, lo + 100); n = hi - lo
            RSb, RBb = RS[idx[lo:hi]], RB[idx[lo:hi]]; ones = np.ones((n, T), bool)
            sym = simulate(RSb, RBb, ones, np.full(n, omega - W / 2), np.full(n, omega + W / 2), np.zeros(n, int),
                           np.full(n, omega), np.full(T, cup), np.full(T, cdn))
            Lb = np.empty((n, T)); Ub = np.empty((n, T))
            for i in range(n):
                Lb[i], Ub[i], _ = d5.dcb_bounds(sym["w"][i], years, omega, W)
            dcb = simulate(RSb, RBb, ones, Lb, Ub, np.zeros(n, int), np.full(n, omega),
                           np.full(T, cup), np.full(T, cdn), keep="min")
            rt = omega * RSb + (1 - omega) * RBb - 1
            ils, ild = il(sym["cost"], sym["r"], rt, 5.0), il(dcb["cost"], dcb["r"], rt, 5.0)
            for i in range(n):
                rows.append(dict(block=blk, path=lo + i, il_sym=ils[i], il_dcb=ild[i]))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "rob_mechanism.csv"), index=False)
    return df


def monthly_detail(seed=31):
    """Month-end monitoring: best-width IL by family (full sample), decomposition
    at each family's best width, and OOS SYM - ASY / SYM - MON with bootstrap CIs."""
    rows = []
    for sp in d5.SPECS:
        tag, dates, RS, RB, omega, cup, cdn, burn = spec_data(sp)
        T = len(RS); rt = omega * RS + (1 - omega) * RB - 1; sh = cup / (cup + cdn)
        mon = calendar_mask(dates, "M"); Wg = d5.WGRID; K = len(Wg); monK = np.tile(mon, (K, 1))
        sym = simulate(RS, RB, monK, omega - Wg / 2, omega + Wg / 2, np.zeros(K, int), np.full(K, omega),
                       np.full(T, cup), np.full(T, cdn))
        asy = simulate(RS, RB, monK, omega - sh * Wg, omega + (1 - sh) * Wg, np.zeros(K, int), np.full(K, omega),
                       np.full(T, cup), np.full(T, cdn))
        ils, ila = il(sym["cost"], sym["r"], rt, 5.0), il(asy["cost"], asy["r"], rt, 5.0)
        out = dict(tag=tag, W_sym=Wg[ils.argmin()], il_sym=ils.min(), W_asy=Wg[ila.argmin()], il_asy=ila.min())
        k10 = int(np.argmin(np.abs(Wg - 0.10)))
        out["il_sym10"], out["il_asy10"] = ils[k10], ila[k10]
        # decomposition ASY - SYM at W = 10% under monthly monitoring
        wprev_s = np.r_[omega, sym["w"][k10, :-1]]; wprev_a = np.r_[omega, asy["w"][k10, :-1]]
        gs = wprev_s * RS + (1 - wprev_s) * RB; ga = wprev_a * RS + (1 - wprev_a) * RB
        out["alloc10"] = ((wprev_a - wprev_s) * (RS - RB)).mean() * TD
        out["cost10"] = -(ga * asy["cost"][k10] - gs * sym["cost"][k10]).mean() * TD
        # OOS: IL-selected widths (gamma 5)
        years = pd.DatetimeIndex(dates).year.values; first = years[0] + burn; oos = years >= first
        cal = simulate(RS, RB, mon[None], np.array([omega]), np.array([omega]), np.ones(1, int), np.array([omega]),
                       np.full(T, cup), np.full(T, cdn))
        ser = {}
        for f, sim, lo_, hi_ in [("SYM", sym, omega - Wg / 2, omega + Wg / 2),
                                 ("ASY", asy, omega - sh * Wg, omega + (1 - sh) * Wg)]:
            Lt = np.empty(T); Ut = np.empty(T)
            for y in np.unique(years):
                m = years == y
                if y < first:
                    j = k10
                else:
                    upto = int(np.argmax(years >= y)); j = int(np.argmin(il(sim["cost"][:, :upto], sim["r"][:, :upto], rt[:upto], 5.0)))
                Lt[m] = lo_[j]; Ut[m] = hi_[j]
            ser[f] = simulate(RS, RB, mon[None], Lt[None], Ut[None], np.zeros(1, int), np.array([omega]),
                              np.full(T, cup), np.full(T, cdn))
        ser["MON"] = cal
        idx = stationary_indices(int(oos.sum()), an.B_PAIRED, 21, np.random.default_rng(seed))
        for a, bb in [("SYM", "ASY"), ("SYM", "MON")]:
            bd = d5.boot_diff(ser[a]["r"][0, oos], ser[bb]["r"][0, oos], rt[oos], ser[a]["cost"][0, oos],
                              ser[bb]["cost"][0, oos], idx)
            est = il(ser[a]["cost"][0, oos], ser[a]["r"][0, oos], rt[oos], 5.0) - il(ser[bb]["cost"][0, oos], ser[bb]["r"][0, oos], rt[oos], 5.0)
            out.update({"oos_%s_%s" % (a, bb): est, "oos_%s_%s_lo" % (a, bb): bd["dil"][0], "oos_%s_%s_hi" % (a, bb): bd["dil"][1]})
        rows.append(out); print("monthly done", tag)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "rob_monthly.csv"), index=False)
    return df


TW_EXT = ("Taiwan extended", "0050.TW", "IEF", 0.6,
          dict(Sbuy=14.25, Ssell=24.25, Bbuy=14.25, Bsell=14.25), "tw", 5)


def taiwan_extended(start="2014-01-03", seed=41):
    """0050 with a currency-hedged US Treasury leg (IEF in USD; hedging carry ignored),
    treated as tax-exempt, 2014-2026.  Starts after Yahoo's mis-applied 0050 split on
    2014-01-02.  (Yahoo's USD/TWD history contains bad quotes in 2011-2016, so an
    unhedged TWD conversion is not used.)"""
    tag, s, b, omega, c, rk, burn = TW_EXT
    dates, RS, RB = an.pair(s, b, start)
    T = len(RS); years = pd.DatetimeIndex(dates).year.values
    cup = (c["Sbuy"] + c["Bsell"]) * BP; cdn = (c["Ssell"] + c["Bbuy"]) * BP; sh = cup / (cup + cdn)
    rt = omega * RS + (1 - omega) * RB - 1
    out = dict(tag=tag, start=str(dates[0].date()), end=str(dates[-1].date()), T=T,
               max_abs_r_S=float(np.abs(RS - 1).max()), max_abs_r_B=float(np.abs(RB - 1).max()))
    # location sweep at W = 10%
    K = len(TH)
    sw = simulate(RS, RB, np.ones((K, T), bool), omega - TH * 0.10, omega + (1 - TH) * 0.10, np.zeros(K, int),
                  np.full(K, omega), np.full(T, cup), np.full(T, cdn), keep="min")
    ilv = il(sw["cost"], sw["r"], rt, 5.0)
    out.update(opt_share=TH[ilv.argmin()], il_centred=ilv[np.isclose(TH, 0.5)][0])
    # decomposition at W = 10%
    sym, asy, dcb, (Lb, Ub, offs) = d5.run_family(RS, RB, years, omega, cup, cdn, d5.WGRID)
    k = int(np.argmin(np.abs(d5.WGRID - 0.10)))
    out["il_theory"] = il(asy["cost"][k], asy["r"][k], rt, 5.0)
    wps = np.r_[omega, sym["w"][k, :-1]]; wpa = np.r_[omega, asy["w"][k, :-1]]
    gs = wps * RS + (1 - wps) * RB; ga = wpa * RS + (1 - wpa) * RB
    out["alloc10"] = ((wpa - wps) * (RS - RB)).mean() * TD
    out["cost10"] = -(ga * asy["cost"][k] - gs * sym["cost"][k]).mean() * TD
    out["avgw_sym"], out["avgw_asy"] = sym["w"][k].mean(), asy["w"][k].mean()
    # OOS (daily monitoring, gamma 5), same convention as 05
    first = years[0] + burn; oos = years >= first
    cal = simulate(RS, RB, np.stack([calendar_mask(dates, "M"), calendar_mask(dates, "A")]), np.full(2, omega),
                   np.full(2, omega), np.ones(2, int), np.full(2, omega), np.full(T, cup), np.full(T, cdn))
    Wg = d5.WGRID; ser = {}
    for f, sim, LL, UU in [("SYM", sym, omega - Wg[:, None] / 2 + 0 * Lb, omega + Wg[:, None] / 2 + 0 * Ub),
                           ("ASY", asy, omega - sh * Wg[:, None] + 0 * Lb, omega + (1 - sh) * Wg[:, None] + 0 * Ub),
                           ("DCB", dcb, Lb, Ub)]:
        Lt = np.empty(T); Ut = np.empty(T)
        for y in np.unique(years):
            m = years == y
            if y < first:
                j = k
            else:
                upto = int(np.argmax(years >= y)); j = int(np.argmin(il(sim["cost"][:, :upto], sim["r"][:, :upto], rt[:upto], 5.0)))
            Lt[m] = LL[j][m]; Ut[m] = UU[j][m]
        ser[f] = simulate(RS, RB, np.ones((1, T), bool), Lt[None], Ut[None], np.zeros(1, int), np.array([omega]),
                          np.full(T, cup), np.full(T, cdn))
    ser["MON"] = {kk: v[0:1] for kk, v in cal.items()}; ser["ANN"] = {kk: v[1:2] for kk, v in cal.items()}
    for f, so in ser.items():
        out["oos_il_" + f] = il(so["cost"][0, oos], so["r"][0, oos], rt[oos], 5.0)
    idx = stationary_indices(int(oos.sum()), an.B_PAIRED, 21, np.random.default_rng(seed))
    for a, bb in [("SYM", "ASY"), ("SYM", "MON")]:
        bd = d5.boot_diff(ser[a]["r"][0, oos], ser[bb]["r"][0, oos], rt[oos], ser[a]["cost"][0, oos],
                          ser[bb]["cost"][0, oos], idx)
        out["oos_%s_%s_lo" % (a, bb)], out["oos_%s_%s_hi" % (a, bb)] = bd["dil"]
    out["oos_start"] = str(dates[oos][0].date())
    df = pd.DataFrame([out]); df.to_csv(os.path.join(RES, "rob_taiwan_extended.csv"), index=False)
    return df


if __name__ == "__main__" and os.environ.get("ONLY_TWEXT") == "1":
    print(taiwan_extended().T); sys.exit(0)

if __name__ == "__main__" and os.environ.get("ONLY_MONTHLY") == "1":
    monthly_detail(); sys.exit(0)

if __name__ == "__main__" and os.environ.get("ONLY_OOS") == "1":
    location_and_oos(); sys.exit(0)

if __name__ == "__main__":
    m = mechanism_resampling()
    print(m.assign(win=m.il_dcb < m.il_sym).groupby("block").win.mean())
    t = theta_boot()
    for tag, g in t.groupby("tag"):
        print(tag, "P(opt>=0.5)=%.3f  2.5-97.5%%: %s  P(theory worse than centred)=%.3f" % (
            (g.opt_share >= 0.5).mean(), np.percentile(g.opt_share, [2.5, 97.5]), (g.il_theory > g.il_centred).mean()))
    location_and_oos()
    monthly_detail()
    print(taiwan_extended().T)
    L = pd.read_csv(os.path.join(RES, "rob_location.csv")); O = pd.read_csv(os.path.join(RES, "rob_oos.csv"))
    print(L.groupby(["monitoring", "gamma"]).agg(opt_min=("opt_share", "min"), opt_max=("opt_share", "max"),
          theory_worse=("il_theory", lambda x: (x > L.loc[x.index, "il_centred"]).sum())))
    print(O.groupby(["monitoring", "gamma"]).agg(sym_best=("best", lambda x: (x == "SYM").sum()),
          red_min=("red_vs_mon", "min"), red_max=("red_vs_mon", "max"), sym_lt_asy=("sym_lt_asy", "sum")))
