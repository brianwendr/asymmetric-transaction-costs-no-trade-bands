# Replication package

**Asymmetric Transaction Costs and No-Trade Bands: Evidence from ETF Allocation Strategies**

## Contents
| Path | Purpose |
|---|---|
| `code/01_download_data.py` | Downloads daily adjusted prices (Yahoo Finance chart endpoint) and the 3-month T-bill rate (FRED `DTB3`). Standard library only. |
| `code/engine.py` | Backtest engine (band, threshold, calendar rules with asymmetric proportional costs), stationary bootstrap. |
| `code/02_analysis.py` | Produces every table, figure input and number reported in the paper. Pre-specified design is documented in its header. |
| `code/03_tables_figure.py` | Formats Tables 1–2 and Figure 1 from `results/`. |
| `code/05_drift_corrected.py` | Location sweep, implementation-loss frontier, drift-corrected band, out-of-sample IL tests, re-simulation. |
| `code/06_robustness.py` | γ and monitoring-frequency robustness, sampling distribution of the optimal location, iid-vs-block mechanism test, extended Taiwanese sample. |
| `code/04_verify_text.py` | Asserts that every number quoted in the manuscript matches `results/`. |
| `data/` | Raw CSVs as downloaded on 20 Oct 2026 (date, close, adjclose). Yahoo Finance terms restrict redistribution, so exclude `data/` from the public GitHub repository and provide the snapshot on request; `01_download_data.py` re-creates it. |
| `results/` | All outputs (CSV/JSON/PNG/TIFF). |

## Reproduce
```bash
pip install numpy pandas matplotlib      # tested: numpy 2.x, pandas 2.x/3.x, matplotlib 3.x
cd code
python3 01_download_data.py              # skip if data/ already holds the 20 Oct 2026 snapshot
python3 02_analysis.py                   # ~7 minutes on a laptop
python3 05_drift_corrected.py            # ~2 minutes
python3 06_robustness.py                 # ~3 minutes
python3 03_tables_figure.py
python3 04_verify_text.py
```
Random seeds are fixed (paired bootstrap seed 1; re-simulation seed 7).
Re-downloading later will extend the sample and Yahoo may revise adjusted prices; set
`END_DATE=2026-09-30` (default) to keep the paper's sample end.

## Data sources
* Yahoo Finance adjusted close (split and distribution adjusted): SPY, IEF, AGG, TLT, VFINX, VUSTX, 0050.TW, 00679B.TWO.
* FRED, Board of Governors of the Federal Reserve System: 3-Month Treasury Bill Secondary Market Rate (DTB3).

## Notes
* The 13-week T-bill yield (`^IRX`, Yahoo) is the US risk-free rate (FRED was unreachable from the download machine; the script uses `FRED_DTB3.csv` automatically if present).
* Yahoo's USD/TWD (`TWD=X`) history contains bad quotes in 2011–2016 (e.g. 3.67 TWD/USD on 2014-12-31), so the extended Taiwanese sample uses a currency-hedged Treasury leg (IEF in USD, hedging carry ignored) instead of an unhedged TWD conversion.
* Yahoo's 0050.TW history before 2014 contains a mis-applied split and implausible prices; the Taiwan sample starts in January 2017 (listing of 00679B) and is unaffected.
* `manuscript/build.sh` rebuilds the Word/PDF manuscript: pandoc (equations become native Word equation objects), then `format_docx.py` (journal layout: A4, Times 12 pt, double spacing, page numbers (no line numbers), three-line tables, hanging-indent references, no keep-with-next markers). `wordcount.py` reports the Word-style count including tables, notes and captions.

## Preparing the public repository

Yahoo Finance's terms restrict redistribution of its price data, so publish the code and keep the
snapshot private:

```bash
printf 'data/\n__pycache__/\n.DS_Store\n*.pyc\n' > .gitignore
git init && git add -A && git commit -m "Replication package"
```

Upload `code/`, `results/`, `README.md` and `requirements.txt`; leave `data/` out and state in the
paper that the snapshot is available from the author on request (as the Data availability statement
already does). 

## Manuscript sources

`manuscript/` holds the paper itself: `body.md` and `tables_figure.md` are the sources,
`build.sh` regenerates `Manuscript_AEL_Wen.docx` and the PDF via pandoc plus `format_docx.py`
(journal layout), and `wordcount.py` reports the word count including tables, notes and captions.
These are for your own editing; they do not need to go in the public repository.
