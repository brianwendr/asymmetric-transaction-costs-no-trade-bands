#!/usr/bin/env python3
"""
03_tables_figure.py -- formats Table 1, Table 2 and Figure 1 from ../results.
Run after 02_analysis.py and 05_drift_corrected.py.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("an", os.path.join(HERE, "02_analysis.py"))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
RES = an.RES
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
ROWS = [("US, psi=50", "US SPY/IEF, ψ = 50"), ("US, psi=200", "US SPY/IEF, ψ = 200"),
        ("Taiwan, statutory", "Taiwan 0050/00679B"), ("US long, psi=50", "US VFINX/VUSTX, ψ = 50")]

# ------------------------------------------------------------------ Table 1
dec = pd.read_csv(os.path.join(RES, "location_decomposition.csv")); dec = dec[dec.W == 0.10].set_index("tag")
sw = pd.read_csv(os.path.join(RES, "location_sweep.csv")); sw = sw[sw.W == 0.10]
full = pd.read_csv(os.path.join(RES, "dcb_full.csv")).set_index("tag")   # exact SYM/ASY at W = 10%
t1 = []
for tag, lab in ROWS:
    r = dec.loc[tag]; g = sw[sw.tag == tag].reset_index(drop=True)
    th = g.theory.iloc[0]; ilc = g[np.isclose(g.theta, 0.5)].il.iloc[0]
    ilt = full.loc[tag, "ASY_il"]; j = g.il.idxmin()
    assert abs(full.loc[tag, "SYM_il"] - ilc) < 1e-12
    t1.append({"Sample": lab, "Lower share (theory)": round(th, 2),
               "Cost comp. (bp)": round(1e4 * r.cost, 2), "Alloc. comp. (bp)": round(1e4 * r.alloc, 1),
               "Avg. w SYM/ASY (%)": "%.1f/%.1f" % (100 * r.avgw_sym, 100 * r.avgw_asy),
               "IL centred (bp)": round(1e4 * ilc, 2), "IL theory shift (bp)": round(1e4 * ilt, 2),
               "IL-optimal share": round(g.theta[j], 2)})
t1 = pd.DataFrame(t1); t1.to_csv(os.path.join(RES, "Table1.csv"), index=False)
print(t1.to_string(index=False))

# ------------------------------------------------------------------ Table 2
o = pd.read_csv(os.path.join(RES, "dcb_oos.csv"))
fams = [("SYM", "Centred band"), ("ASY", "Cost-shifted band"), ("DCB", "Drift-corrected band"),
        ("TGT", "Threshold-to-target"), ("MON", "Monthly rebalancing"), ("ANN", "Annual rebalancing")]
t2 = []
for f, lab in fams:
    row = {"Strategy": lab}
    for tag, cl in ROWS:
        x = o[(o.tag == tag) & (o.family == f)].iloc[0]
        row[cl] = "%.2f" % (1e4 * x.il) + ("" if np.isnan(x.median_param) else " (%.2f)" % x.median_param)
    t2.append(row)
t2 = pd.DataFrame(t2); t2.to_csv(os.path.join(RES, "Table2.csv"), index=False)
print(t2.to_string(index=False))

# ------------------------------------------------------------------ CE-based OOS tests, ASY vs SYM (Section 4.3)
allt = []
for tag in ["US_psi10", "US_psi25", "US_psi50", "US_psi100", "US_psi200", "TW_statutory",
            "US_AGG_psi50", "US_TLT_psi50", "US_LONG_psi50", "US_w40_psi50", "US_w80_psi50",
            "US_monthly_psi50", "US_stress"]:
    tt = pd.read_csv(os.path.join(RES, "%s_tests.csv" % tag))
    tt = tt[(tt.a == "ASY") & (tt.b == "SYM")].iloc[0]
    allt.append(dict(tag=tag, dce_bp=1e4 * tt.dce, lo=1e4 * tt.dce_lo, hi=1e4 * tt.dce_hi, p=tt.p_ce,
                     dsharpe=tt.dsharpe, p_sharpe=tt.p_sharpe))
pd.DataFrame(allt).to_csv(os.path.join(RES, "OOS_ASY_vs_SYM_all.csv"), index=False)

# ------------------------------------------------------------------ Figure 1
psis = [10, 25, 50, 100, 200]
US = dict(Sbuy=5, Bsell=5, Bbuy=5)
d = pd.DataFrame([an.decomposition("psi%d" % p, "SPY", "IEF", 0.6, dict(US, Ssell=5 + p), W=0.10, B=1000)
                  for p in psis])
d.to_csv(os.path.join(RES, "Figure1a_data.csv"), index=False)
sw.to_csv(os.path.join(RES, "Figure1b_data.csv"), index=False)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), dpi=150)
a = ax[0]; x = np.arange(len(psis))
a.plot(x, 1e4 * d.alloc, "-o", color=BLUE, lw=2, ms=6, label="Allocation (tilt) component")
a.plot(x, 1e4 * d.cost, "-s", color=ORANGE, lw=2, ms=6, label="Cost component")
a.set_yscale("log"); a.set_ylim(0.004, 300)
a.set_xticks(x); a.set_xticklabels([str(p) for p in psis])
a.set_xlabel("Sell-side equity surcharge ψ (bp)")
a.set_ylabel("ASY − SYM, bp per year (log scale)")
a.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=8)
a.set_title("(a) Return difference, SPY/IEF, W = 10%", fontsize=9, loc="left", color=INK)
b = ax[1]
for (tag, lab), col, mk in zip([ROWS[0], ROWS[1], ROWS[2]], [BLUE, ORANGE, AQUA], ["o", "s", "^"]):
    g = sw[sw.tag == tag].sort_values("theta")
    b.plot(g.theta, 1e4 * g.il, "-", marker=mk, color=col, lw=2, ms=4.5, label=lab)
    th = g.theory.iloc[0]
    b.plot([th], [1e4 * full.loc[tag, "ASY_il"]], marker=mk, ms=9, mfc="white", mec=col, mew=2, ls="none")
b.axvline(0.5, color=MUTED, lw=0.8, ls=":")
b.set_ylim(0, 10); b.set_xlim(0, 0.95)
b.set_xlabel("Share of band width below target")
b.set_ylabel("Implementation loss (bp per year)")
b.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.62, 1.0))
b.set_title("(b) Loss by band location, W = 10%", fontsize=9, loc="left", color=INK)
for axx in ax:
    axx.grid(axis="y", color=GRID, lw=0.6); axx.set_axisbelow(True)
    for s in ("top", "right"):
        axx.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(RES, "Figure1.png"), dpi=600)
fig.savefig(os.path.join(RES, "Figure1.tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("Figure saved")
