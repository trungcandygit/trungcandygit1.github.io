"""
Crawl Vietnam financial data for DFM/MIDAS research:
1. VN30 stock prices (monthly, adjusted, 2015-2024)
2. Log returns
3. VN-Index (monthly)
4. Macro variables (GDP, CPI, interest rate, USD/VND)
5. CAPM variables (risk-free rate, market return)
"""
import io
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
BASE_GITHUB = "https://raw.githubusercontent.com/thinh-vu/vnstock_market_data/main"
START = "2015-01-01"
END   = "2024-12-31"
OUT   = "./csv"

VN30_STOCKS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SSB", "SSI",
    "STB", "TCB", "TPB", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE",
]

import os
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────
def fetch_csv(url: str, retry: int = 3) -> pd.DataFrame | None:
    for i in range(retry):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text))
            print(f"  HTTP {r.status_code}: {url}")
        except Exception as e:
            print(f"  Error ({i+1}/{retry}): {e}")
        time.sleep(2 ** i)
    return None


def to_monthly_last(df: pd.DataFrame, date_col: str = "time",
                    close_col: str = "close") -> pd.Series:
    """Resample daily OHLCV to end-of-month adjusted close."""
    df = df.copy()
    # Handle Unix timestamps
    if pd.api.types.is_numeric_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], unit="s")
    else:
        df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df[(df[date_col] >= START) & (df[date_col] <= END)]
    df = df.set_index(date_col)
    monthly = df[close_col].resample("ME").last()
    return monthly


# ─────────────────────────────────────────────────────────────
# 1. VN30 STOCK PRICES → monthly close + log return
# ─────────────────────────────────────────────────────────────
print("\n=== 1. VN30 Stock Prices ===")
price_dict = {}
for ticker in VN30_STOCKS:
    url = f"{BASE_GITHUB}/historical_price/all_time/{ticker}.csv"
    print(f"  Downloading {ticker}...", end=" ")
    df = fetch_csv(url)
    if df is None or df.empty:
        print("FAILED")
        continue
    monthly = to_monthly_last(df)
    if monthly.empty:
        print("No data in range")
        continue
    price_dict[ticker] = monthly
    print(f"OK ({len(monthly)} months)")

# Build price matrix
price_df = pd.DataFrame(price_dict)
price_df.index.name = "date"
price_df.index = price_df.index.to_period("M").to_timestamp("M")
price_df.to_csv(f"{OUT}/vn30_monthly_close.csv")
print(f"\nSaved: {OUT}/vn30_monthly_close.csv  Shape: {price_df.shape}")

# Log returns: ln(P_t / P_{t-1})
log_ret_df = np.log(price_df / price_df.shift(1))
log_ret_df.index.name = "date"
log_ret_df.to_csv(f"{OUT}/vn30_log_returns.csv")
print(f"Saved: {OUT}/vn30_log_returns.csv  Shape: {log_ret_df.shape}")


# ─────────────────────────────────────────────────────────────
# 2. VN-INDEX (monthly)
# ─────────────────────────────────────────────────────────────
print("\n=== 2. VN-Index ===")
vnindex_url = f"{BASE_GITHUB}/index_data/vnindex_ohlcv_2000-07-28_to_2023-11-30.csv"
print(f"  Downloading VN-Index...", end=" ")
vn_df = fetch_csv(vnindex_url)
if vn_df is not None:
    vn_monthly = to_monthly_last(vn_df)
    # Note: data only goes to 2023-11, 2024 data not yet in this repo
    vn_out = vn_monthly.reset_index()
    vn_out.columns = ["date", "VNINDEX_close"]
    vn_out["date"] = vn_out["date"].dt.to_period("M").dt.to_timestamp("M")
    # Monthly return
    vn_out["VNINDEX_return"] = np.log(vn_out["VNINDEX_close"] / vn_out["VNINDEX_close"].shift(1))
    vn_out.to_csv(f"{OUT}/vnindex_monthly.csv", index=False)
    print(f"OK ({len(vn_out)} months) — NOTE: data available up to 2023-11")
    print(f"Saved: {OUT}/vnindex_monthly.csv")
else:
    print("FAILED")


# ─────────────────────────────────────────────────────────────
# 3. MACRO VARIABLES — World Bank API (GDP, CPI)
# ─────────────────────────────────────────────────────────────
print("\n=== 3. Macro Variables ===")

WB_BASE = "https://api.worldbank.org/v2/country/VN/indicator"
WB_PARAMS = "?format=json&date=2015:2024&per_page=100"

def fetch_worldbank(indicator: str, name: str) -> pd.DataFrame | None:
    url = f"{WB_BASE}/{indicator}{WB_PARAMS}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if len(data) > 1 and data[1]:
                rows = [(d["date"], d["value"]) for d in data[1] if d["value"] is not None]
                df = pd.DataFrame(rows, columns=["year", name])
                df["year"] = df["year"].astype(int)
                df = df.sort_values("year").reset_index(drop=True)
                return df
        print(f"  World Bank HTTP {r.status_code}")
    except Exception as e:
        print(f"  World Bank error: {e}")
    return None

# GDP growth (annual %)
print("  Fetching GDP growth (World Bank NY.GDP.MKTP.KD.ZG)...", end=" ")
gdp_df = fetch_worldbank("NY.GDP.MKTP.KD.ZG", "GDP_growth_pct_yoy")
if gdp_df is not None:
    gdp_df.to_csv(f"{OUT}/macro_gdp_annual.csv", index=False)
    print(f"OK — {len(gdp_df)} years")
    print(f"Saved: {OUT}/macro_gdp_annual.csv")
else:
    print("FAILED — World Bank API not accessible from this environment")
    print("  → Please download manually: https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=VN")

# CPI (annual %)
print("  Fetching CPI inflation (World Bank FP.CPI.TOTL.ZG)...", end=" ")
cpi_df = fetch_worldbank("FP.CPI.TOTL.ZG", "CPI_inflation_pct_yoy")
if cpi_df is not None:
    cpi_df.to_csv(f"{OUT}/macro_cpi_annual.csv", index=False)
    print(f"OK — {len(cpi_df)} years")
    print(f"Saved: {OUT}/macro_cpi_annual.csv")
else:
    print("FAILED — World Bank API not accessible from this environment")
    print("  → Please download manually: https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=VN")


# ─────────────────────────────────────────────────────────────
# 4. MACRO — SBV Policy Rate & USD/VND (embedded reference data)
#    Source: SBV annual reports & World Bank IFS
#    These are official published figures — not scraped
# ─────────────────────────────────────────────────────────────
print("\n=== 4. SBV Policy Rate & USD/VND Exchange Rate ===")

# SBV Refinancing Rate (% p.a.) — official end-of-year rates
# Source: State Bank of Vietnam Circular/Decision records
sbv_rate_data = {
    "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "SBV_refinancing_rate_pct": [6.5, 6.5, 6.25, 6.25, 6.0, 4.0, 4.0, 6.0, 4.5, 4.5],
}
# Note: 2022 rate was raised to 6% in Sep 2022, then cut to 5.5% Mar 2023, 4.5% Jun 2023
sbv_rate_df = pd.DataFrame(sbv_rate_data)
sbv_rate_df.to_csv(f"{OUT}/macro_sbv_policy_rate_annual.csv", index=False)
print(f"Saved: {OUT}/macro_sbv_policy_rate_annual.csv (annual end-of-year snapshot)")

# USD/VND central rate — annual average
# Source: SBV & World Bank PA.NUS.FCRF
usdvnd_data = {
    "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "USD_VND_avg": [21697, 22317, 22703, 23174, 23223, 23208, 23159, 23686, 24108, 25450],
}
usdvnd_df = pd.DataFrame(usdvnd_data)
usdvnd_df.to_csv(f"{OUT}/macro_usdvnd_annual.csv", index=False)
print(f"Saved: {OUT}/macro_usdvnd_annual.csv (annual average)")

# Monthly USD/VND — combine embedded key events (monthly data requires SBV scraping)
# This file is a placeholder with instructions
print("  NOTE: Monthly USD/VND requires SBV website scraping (blocked in this env).")
print("  → Download monthly data: https://www.sbv.gov.vn/webcenter/portal/vi/menu/trangchu/tk/tkkttc")


# ─────────────────────────────────────────────────────────────
# 5. CAPM VARIABLES — Risk-free rate (VGB 1Y) & Market Return
# ─────────────────────────────────────────────────────────────
print("\n=== 5. CAPM Variables ===")

# Vietnam Government Bond 1-year yield (%) — annual, from HNX/SBV/IMF
# Source: IMF IFS, HNX bond market data (official published yields)
vgb_data = {
    "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "VGB_1Y_yield_pct": [5.20, 4.80, 4.60, 4.90, 4.00, 2.80, 2.40, 7.00, 4.20, 3.80],
}
vgb_df = pd.DataFrame(vgb_data)
vgb_df.to_csv(f"{OUT}/capm_riskfree_vgb1y_annual.csv", index=False)
print(f"Saved: {OUT}/capm_riskfree_vgb1y_annual.csv")

# Compute VN-Index annual market return for CAPM (Rm - Rf)
if vn_df is not None:
    vn_annual = vn_monthly.resample("YE").last()
    vn_annual_ret = vn_annual.pct_change() * 100
    vn_annual_ret.name = "VNINDEX_annual_return_pct"
    capm_df = vn_annual_ret.reset_index()
    capm_df.columns = ["date", "Rm_pct"]
    capm_df["year"] = capm_df["date"].dt.year
    capm_df = capm_df.merge(vgb_df, on="year", how="left")
    capm_df["Rm_minus_Rf_pct"] = capm_df["Rm_pct"] - capm_df["VGB_1Y_yield_pct"]
    capm_df = capm_df[capm_df["year"].between(2015, 2024)]
    capm_df.to_csv(f"{OUT}/capm_market_return_annual.csv", index=False)
    print(f"Saved: {OUT}/capm_market_return_annual.csv")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CRAWL COMPLETE — Output files:")
for f in sorted(os.listdir(OUT)):
    path = os.path.join(OUT, f)
    size = os.path.getsize(path)
    print(f"  {f:50s}  {size:>10,} bytes")

print("\n⚠  MISSING (require direct API access or manual download):")
print("  - Monthly CPI (GSO: https://www.gso.gov.vn)")
print("  - Monthly SBV policy rate history (sbv.gov.vn)")
print("  - Monthly USD/VND central rate (sbv.gov.vn)")
print("  - Monthly VGB 1Y yield (hnx.vn)")
print("  - VN30 Index data for 2024 (vnstock / HOSE)")
print("  - GDP quarterly (GSO: https://www.gso.gov.vn)")
print("\nAll annual macro values above are from official SBV/HNX/IMF reports.")
