#!/usr/bin/env python3
"""04_verify_text.py -- asserts that every number quoted in the manuscript text
matches ../results.  Run after 02, 05 and 03."""
import os, json, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rd = lambda f: pd.read_csv(os.path.join(R, f))
dec = rd("location_decomposition.csv"); d10 = dec[dec.W == 0.10].set_index("tag").drop("US, psi=0")
full = rd("dcb_full.csv").set_index("tag")
sw = rd("location_sweep.csv"); sw10 = sw[sw.W == 0.10]
oos = rd("dcb_oos.csv"); tests = rd("dcb_oos_tests.csv")
rs = rd("US_resim_W10.csv"); dres = rd("dcb_resim.csv")
mech = rd("mechanism.csv"); ce_oos = rd("OOS_ASY_vs_SYM_all.csv")
checks = []
def chk(name, val, target, tol):
    checks.append((abs(val - target) <= tol, name, float(val), target))

u = d10.loc["US, psi=50"]
# abstract / introduction
chk("cost saving < 0.5 bp (max)", 1e4 * d10.cost.max(), 0.41, 0.005)
chk("avg weight up to 4.5 pp", 100 * (d10.avgw_asy - d10.avgw_sym).max(), 4.5, 0.05)
chk("tilt 18 bp (min)", 1e4 * d10.alloc.min(), 17.7, 0.05)
chk("tilt 43 bp (max)", 1e4 * d10.alloc.max(), 43.2, 0.05)
exc = 100 * (full.ASY_il / full.SYM_il - 1)
chk("IL excess min 16%", exc.min(), 16, 0.5); chk("IL excess max 120%", exc.max(), 120, 0.5)
p = oos.pivot(index="tag", columns="family", values="il")
red = 100 * (1 - p.SYM / p.MON)
chk("OOS reduction vs monthly min 31%", red.min(), 31, 0.5); chk("max 52%", red.max(), 52, 0.5)
best = p[["SYM", "ASY", "DCB", "TGT", "MON", "ANN", "BH"]].idxmin(axis=1)
chk("SYM lowest in 7 of 8", (best == "SYM").sum(), 7, 0)
# section 4.1
chk("daily 19.5", u.cost_daily_bp, 19.5, 0.05); chk("monthly 4.1", u.cost_monthly_bp, 4.1, 0.05)
chk("band 1.33", u.cost_sym_bp, 1.33, 0.005); chk("90% of days above", 100 * u.frac_upper_sym, 90, 0.5)
chk("63.2", 100 * u.avgw_sym, 63.2, 0.05); chk("66.8", 100 * u.avgw_asy, 66.8, 0.05)
chk("0.08", 1e4 * u.cost, 0.08, 0.005); chk("CI 0.04", 1e4 * u.cost_lo, 0.04, 0.005); chk("CI 0.12", 1e4 * u.cost_hi, 0.12, 0.005)
chk("34.2", 1e4 * u.alloc, 34.2, 0.05)
g = rs[rs.psi == 50]; chk("CE win 37%", 100 * (g.dce > 0).mean(), 37, 0.5)
b = np.polyfit(g.spread, g.dce, 1); chk("breakeven 10.2%", -100 * b[1] / b[0], 10.2, 0.05)
# section 4.2
opt = sw10.loc[sw10.groupby("tag").il.idxmin()].set_index("tag")
chk("optimal share min 0.55", opt.theta.min(), 0.55, 1e-9); chk("optimal share max 0.70", opt.theta.max(), 0.70, 1e-9)
chk("optimum never on cost side", float((opt.theta < 0.5).sum()), 0, 0)
cen = sw10[np.isclose(sw10.theta, 0.5)].set_index("tag").il
us = [t for t in opt.index if t.startswith("US")]
chk("US within 0.15 bp of centred", 1e4 * (cen[us] - opt.il[us]).max(), 0.15, 0.005)
chk("below-target share min 5%", 100 * mech.share_below.min(), 5, 0.5)
chk("below-target share max 19%", 100 * mech.share_below.max(), 19, 0.5)
chk("ratio min 2.5", mech.ratio.min(), 2.45, 0.05); chk("ratio max 10", mech.ratio.max(), 10.2, 0.05)
pv = dres.pivot(index="path", columns="family", values="il")
chk("SYM beats ASY in 99.9% of histories", 100 * (pv.SYM < pv.ASY).mean(), 99.9, 0.05)
# section 4.3
t = tests.assign(ex=(tests.dil_hi < 0) | (tests.dil_lo > 0))
for bb, neg, sig in [("MON", 8, 8), ("ASY", 8, 6)]:
    x = t[t.b == bb]; chk("SYM<%s in %d" % (bb, neg), (x.dil < 0).sum(), neg, 0); chk("CI excl. 0 vs %s: %d" % (bb, sig), x.ex.sum(), sig, 0)
chk("DCB raises loss in 7 of 8", (t[t.b == "DCB"].dil < 0).sum(), 7, 0)
chk("Holm min p 0.57", min(1, ce_oos.p.min() * len(ce_oos)), 0.57, 0.005); chk("13 specs", len(ce_oos), 13, 0)
mp = oos[oos.family == "SYM"].set_index("tag").median_param
psi50 = [i for i in mp.index if "psi=50" in i]
chk("width psi=50 min 5%", mp[psi50].min(), 0.05, 1e-9); chk("width psi=50 max 7%", mp[psi50].max(), 0.07, 1e-9)
chk("width psi=200 11%", mp["US, psi=200"], 0.11, 1e-9)
dcb_share = (0.05 + full.final_offset) / 0.10          # lower share implied by the DCB's final offset at W = 10%
chk("DCB overshoots optimum in all 8", float((dcb_share.values > opt.loc[full.index].theta.values).all()), 1, 0)
# v3 additions: robustness (06)
rl = rd("rob_location.csv"); ro = rd("rob_oos.csv"); rm = rd("rob_monthly.csv")
rb = rd("rob_theta_boot.csv"); rmech = rd("rob_mechanism.csv"); tx = rd("rob_taiwan_extended.csv").iloc[0]
chk("48/48 ASY worse at W=10%", float((rl.il_theory > rl.il_centred).sum()), 48, 0)
chk("48 combos", len(rl), 48, 0)
dd = rl[rl.monitoring == "daily"]
chk("daily opt share >= 0.5 (centred or down)", float((dd.opt_share >= 0.5).all()), 1, 0)
mm = rl[rl.monitoring == "monthly"]
chk("monthly opt share min 0.40", mm.opt_share.min(), 0.40, 1e-9); chk("monthly opt share max 0.80", mm.opt_share.max(), 0.80, 1e-9)
for g in (2.0, 10.0):
    x = ro[(ro.monitoring == "daily") & (ro.gamma == g)]
    chk("gamma %g: SYM<ASY 8/8" % g, float(x.sym_lt_asy.sum()), 8, 0)
x = ro[(ro.monitoring == "daily") & ro.gamma.isin([2.0, 10.0])]
chk("gamma 2,10 reduction min 27%", 100 * x.red_vs_mon.min(), 27, 0.5); chk("max 57%", 100 * x.red_vs_mon.max(), 57, 0.5)
x5 = ro[(ro.monitoring == "daily") & (ro.gamma == 5.0)]
chk("06 daily g5 consistent with 05: SYM best 7", float((x5.best == "SYM").sum()), 7, 0)
sig = ((rm.oos_SYM_ASY_lo > 0) | (rm.oos_SYM_ASY_hi < 0)).sum()
chk("monthly: ASY vs SYM significant in 1/8", float(sig), 1, 0)
pw = rb.groupby("tag").apply(lambda g: (g.il_theory > g.il_centred).mean())
chk("theta boot US 100%", 100 * pw["US, psi=50"], 100, 0.05); chk("theta boot TW 99.8%", 100 * pw["Taiwan, statutory"], 99.8, 0.05)
mw = rmech.assign(w=rmech.il_dcb < rmech.il_sym).groupby("block").w.mean()
chk("DCB win block 25%", 100 * mw[63], 25, 0.5); chk("DCB win iid 45%", 100 * mw[1], 45, 0.5)
chk("TW ext cost 0.04", 1e4 * tx.cost10, 0.04, 0.005); chk("TW ext tilt 14.6", 1e4 * tx.alloc10, 14.6, 0.05)
chk("TW ext IL +14%", 100 * (tx.il_theory / tx.il_centred - 1), 14, 0.5)
chk("TW ext SYM 41% below MON", 100 * (1 - tx.oos_il_SYM / tx.oos_il_MON), 41, 0.5)
chk("TW ext SYM vs MON CI excl 0", float(tx.oos_SYM_MON_hi < 0), 1, 0)
chk("TW ext DCB best", float(min(["SYM", "ASY", "DCB", "MON", "ANN"], key=lambda f: tx["oos_il_" + f]) == "DCB"), 1, 0)
chk("TW short: DCB best", float(best["Taiwan, statutory"] == "DCB"), 1, 0)
# Table 1 consistency: cost-component CIs exclude zero
t1rows = ["US, psi=50", "US, psi=200", "Taiwan, statutory", "US long, psi=50"]
chk("Table 1 cost CIs exclude 0", float((d10.loc[t1rows].cost_lo > 0).all()), 1, 0)
bad = [c for c in checks if not c[0]]
for ok, n, v, tt in checks:
    print("%s  %-40s %10.4f  (text %s)" % ("OK " if ok else "BAD", n, v, tt))
print("\n%d checks, %d failed" % (len(checks), len(bad)))
