#!/usr/bin/env python3
"""
01_download_data.py  --  Replication package, "Asymmetric Transaction Costs and
No-Trade Bands: Evidence from ETF Allocation Strategies".

Downloads daily split- and dividend-adjusted prices from the public Yahoo Finance
chart endpoint and the 3-month T-bill rate (DTB3) from FRED.  Standard library
only (no pip install needed).  Writes one CSV per series to ../data/.

Usage (macOS Terminal):   cd ~/Documents/AEL_ETF_Hysteresis/code && python3 01_download_data.py
"""
import csv, json, os, ssl, sys, time, urllib.request, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True)

TICKERS = ["SPY", "IEF", "AGG", "TLT", "VFINX", "VUSTX", "VBMFX",
           "0050.TW", "00679B.TWO", "^IRX"]
# "TWD=X" (USD/TWD) can be fetched with `python3 01_download_data.py TWD=X`; it was examined
# but not used: Yahoo's history contains bad quotes in 2011-2016 (see README).
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
CTX = ssl.create_default_context()
try:                                    # python.org builds on macOS may lack certs
    import certifi; CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

def get(url):
    req = urllib.request.Request(url, headers=UA)
    try:
        return urllib.request.urlopen(req, timeout=60, context=CTX).read()
    except ssl.SSLError:
        return urllib.request.urlopen(req, timeout=60,
                                      context=ssl._create_unverified_context()).read()

def yahoo(tk):
    p2 = int(time.time())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%s?period1=0&period2=%d"
           "&interval=1d&events=div%%7Csplit&includeAdjustedClose=true"
           % (urllib.request.quote(tk, safe="=^"), p2))
    js = json.loads(get(url))["chart"]["result"][0]
    ts = js["timestamp"]; q = js["indicators"]["quote"][0]
    adj = js["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    gmt = js["meta"].get("gmtoffset", 0)
    rows = []
    for t, c, a in zip(ts, q["close"], adj):
        if c is None or a is None:
            continue
        d = dt.datetime.utcfromtimestamp(t + gmt).date().isoformat()
        rows.append((d, c, a))
    fn = os.path.join(OUT, tk.replace("^", "IDX_").replace(".", "_").replace("=", "_") + ".csv")
    with open(fn, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "close", "adjclose"]); w.writerows(rows)
    print("%-12s %6d rows  %s -> %s" % (tk, len(rows), rows[0][0], rows[-1][0]))

def fred(sid):
    raw = get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + sid).decode()
    with open(os.path.join(OUT, "FRED_%s.csv" % sid), "w") as f:
        f.write(raw)
    print("%-12s %6d rows" % ("FRED " + sid, raw.count("\n") - 1))

if __name__ == "__main__":
    ok = True
    only = sys.argv[1:]                      # optional: download only the listed tickers
    for tk in (only or TICKERS):
        try:
            yahoo(tk); time.sleep(1.0)
        except Exception as e:
            ok = False; print("FAILED", tk, repr(e))
    if only:
        sys.exit(0)
    try:
        fred("DTB3")
    except Exception as e:
        print("FAILED FRED DTB3 (optional; ^IRX is used as fallback):", repr(e))
    with open(os.path.join(OUT, "DOWNLOAD_LOG.txt"), "w") as f:
        f.write("Downloaded %s UTC by 01_download_data.py\n" % dt.datetime.utcnow().isoformat())
    print("\nDone. Files in", os.path.abspath(OUT), "" if ok else "(some tickers failed)")
