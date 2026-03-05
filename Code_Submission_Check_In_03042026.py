#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:55:17 2026

@author: amandameulemans
"""

"""
DS785 Capstone – Data Collection, Cleaning, and Preprocessing (Presentation 2)
Project: Wisconsin Statewide Deer Harvest Forecasting

This script builds a modeling-ready master dataset by:
1) Loading historical statewide harvest data (1966–2025) from a CSV export
2) Cleaning and standardizing numeric fields
3) Filling known missing weapon harvest splits for 2014–2015 and 2016–2025 (manual entry from WI DNR tables)
4) Adding engineered time-aware features (lags, YoY change, per-hunter rate)
5) Adding external covariates (Winter Severity Index, post-hunt population estimates) via manual mappings
6) Producing summary diagnostics and saving the final dataset
7) Producing Presentation 2 visuals (saved to disk)

AI USE NOTE:
Some code structure, refactoring, and helper functions were revised with assistance from an AI tool.
All data values used for manual fills (weapon totals, WSI, population) must be verified against the WI DNR sources
and/or your notes before final submission.
 
I used ChatGPT to help revise/debug portions of this script, mainly:
(1) dtype handling (pandas nullable Int64),
(2) safe numeric coercion and cleaning,
(3) feature engineering structure (lag features),
(4) plot saving workflow.
- All final decisions, verification of values, and dataset assembly were performed by me.
- Any manually entered values were pulled from Wisconsin DNR / DeerMetrics sources.


Author: Kyle Meulemans
"""

# =============================
# Imports
# =============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================
# User paths (EDIT THESE)
# =============================
INPUT_CSV = "/Users/amandameulemans/Desktop/DS785/Data/wi_statewide_deer_harvest_master_fixed.csv"
OUT_DATASET = "/Users/amandameulemans/Desktop/DS785/Data/wi_statewide_deer_master_modelready.csv"
FIG_DIR = "/Users/amandameulemans/Desktop/DS785/Figures"

os.makedirs(FIG_DIR, exist_ok=True)

# =============================
# Helper functions
# =============================
def save_show(fig_name: str) -> None:
    """Save the current matplotlib figure and close it."""
    out_path = os.path.join(FIG_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved figure:", out_path)

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Remove commas/whitespace and coerce to numeric."""
    s = (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.strip()
         .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(s, errors="coerce")

def to_int64_safe(df: pd.DataFrame, cols: list[str]) -> None:
    """Convert columns to nullable Int64 where possible."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].round().astype("Int64")

# =============================
# Step 1: Load + fix headers
# =============================
df = pd.read_csv(INPUT_CSV)
df.columns = df.iloc[0]
df = df.drop(0).reset_index(drop=True)
df.columns = [str(c).strip() for c in df.columns]

# Guardrails
required_cols = {"year", "total_harvest", "total_antlered", "total_antlerless", "data_source"}
missing_required = required_cols - set(df.columns)
if missing_required:
    raise ValueError(f"Missing required columns in input CSV: {missing_required}")

# =============================
# Step 2: Clean numeric columns safely
# =============================
non_numeric_cols = ["data_source", "is_preliminary"]
numeric_cols = [c for c in df.columns if c not in non_numeric_cols]

for c in numeric_cols:
    df[c] = clean_numeric_series(df[c])

# is_preliminary handling (optional)
if "is_preliminary" in df.columns:
    df["is_preliminary"] = (
        df["is_preliminary"]
          .astype(str)
          .str.strip()
          .str.lower()
          .replace({"true": True, "false": False, "nan": False, "none": False, "": False})
          .astype(bool)
    )
else:
    df["is_preliminary"] = False

# Year as int and sort
df["year"] = df["year"].astype(int)
df = df.sort_values("year").reset_index(drop=True)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# =============================
# Step 3: Fill weapon harvest splits (manual entry)
# =============================
# 3A) 2016–2025 (Gun + Bow + Crossbow combined into Archery)
recent_weapon_data = {
    2016: {"gun_antlered":105186,"gun_antlerless":123540,"archery_antlered":28172+23562,"archery_antlerless":20100+16214},
    2017: {"gun_antlered":105598,"gun_antlerless":122047,"archery_antlered":25808+27406,"archery_antlerless":19358+19822},
    2018: {"gun_antlered":112443,"gun_antlerless":135171,"archery_antlered":21676+25956,"archery_antlerless":18729+21268},
    2019: {"gun_antlered":83496,"gun_antlerless":112639,"archery_antlered":24376+30004,"archery_antlerless":17752+21953},
    2020: {"gun_antlered":93054,"gun_antlerless":132218,"archery_antlered":27136+37545,"archery_antlerless":20700+28186},
    2021: {"gun_antlered":91843,"gun_antlerless":117445,"archery_antlered":24328+36423,"archery_antlerless":15405+22985},
    2022: {"gun_antlered":107191,"gun_antlerless":133980,"archery_antlered":21446+34804,"archery_antlerless":16571+25628},
    2023: {"gun_antlered":93398,"gun_antlerless":115430,"archery_antlered":21636+33167,"archery_antlerless":14557+21782},
    2024: {"gun_antlered":97721,"gun_antlerless":124916,"archery_antlered":24274+39710,"archery_antlerless":15100+24919},
    2025: {"gun_antlered":96990,"gun_antlerless":134174,"archery_antlered":26016+43295,"archery_antlerless":16766+28528},
}

for year, values in recent_weapon_data.items():
    mask = df["year"] == year
    for col, val in values.items():
        if col not in df.columns:
            df[col] = np.nan
        df.loc[mask, col] = val

# 3B) 2014–2015 (manual entry; verify against your WI DNR source)
# IMPORTANT: You had two different 2014/2015 sets in your draft. Use ONE. This is the one that produced
# consistent totals in your latest run (gun_total 153,736 and 177,482; archery_total 37,814 and 35,440).
weapon_2014_2015 = {
    2014: {"gun_antlered": 71130, "gun_antlerless": 82606, "archery_antlered": 19206, "archery_antlerless": 18608},
    2015: {"gun_antlered": 76880, "gun_antlerless":100602, "archery_antlered": 22487, "archery_antlerless": 12953},
}

for year, values in weapon_2014_2015.items():
    mask = df["year"] == year
    for col, val in values.items():
        if col not in df.columns:
            df[col] = np.nan
        df.loc[mask, col] = val

# Recompute weapon totals
for col in ["gun_total", "archery_total"]:
    if col not in df.columns:
        df[col] = np.nan

df["gun_total"] = df["gun_antlered"] + df["gun_antlerless"]
df["archery_total"] = df["archery_antlered"] + df["archery_antlerless"]

print("\nWeapon totals check (2014–2015):")
print(df[df["year"].isin([2014, 2015])][["year","gun_total","archery_total","total_harvest"]])

print("\nWeapon totals check (2016–2025):")
print(df[df["year"] >= 2016][["year","gun_total","archery_total","total_harvest"]].head(10))

# =============================
# Step 4: External covariates (manual mapping)
# =============================
# Winter Severity Index (WSI)
wsi_data = {
    1960:39,1961:88,1962:60,1963:45,1964:92,1965:39,1966:99,1967:38,1968:116,1969:115,
    1970:113,1971:40,1972:52,1973:76,1974:85,1975:60,1976:66,1977:118,1978:43,1979:37,
    1980:104,1981:67,1982:35,1983:45,1984:102,1985:49,1986:14,1987:80,1988:38,1989:37,
    1990:50,1991:44,1992:48,1993:127,1994:115,1995:16,1996:45,1997:37,1998:83,1999:29,
    2000:47,2001:61,2002:49,2003:36,2004:33,2005:70,2006:60,2007:39,2008:48,2009:49,
    2010:149,2011:50,2012:22,2013:30,2014:66,2015:75,2016:64,2017:22,2018:55,2019:69,
    2020:10,2021:32
}

# Post-hunt population estimates
population_data = {
    2007:1230100,
    2008:1002300,
    2009:987300,
    2010:1161300,
    2011:1150900,
    2012:1280400,
    2013:1125900,
    2014:1093600,
    2015:1181400,
    2016:1345000,
    2017:1377100,
    2018:1510400,
    2019:1311100,
    2020:1611000,
    2021:1554400,
    2022:1669000,
    2023:1628500,
    2024:1825000
}

df["winter_severity_index"] = df["year"].map(wsi_data)
df["post_hunt_population"] = df["year"].map(population_data)

# WSI lag (your note: likely affects next-year harvest)
df["wsi_lag1"] = df["winter_severity_index"].shift(1)

print("\nExternal features coverage:")
print("WSI non-null:", df["winter_severity_index"].notna().sum(), "of", len(df))
print("Population non-null:", df["post_hunt_population"].notna().sum(), "of", len(df))

# =============================
# Step 5: Feature engineering (time-aware)
# =============================
# NOTE: licensed hunter fields are still missing for 2014–2025 in your current dataset.
# We keep these features, but they will be NaN until you add licensed hunter counts.
if "gun_licensed_hunters" not in df.columns:
    df["gun_licensed_hunters"] = np.nan
if "archery_licensed_hunters" not in df.columns:
    df["archery_licensed_hunters"] = np.nan

df["licensed_hunters_total"] = df["gun_licensed_hunters"] + df["archery_licensed_hunters"]

df["harvest_per_hunter"] = np.where(
    df["licensed_hunters_total"] > 0,
    df["total_harvest"] / df["licensed_hunters_total"],
    np.nan
)

df["total_harvest_lag1"] = df["total_harvest"].shift(1)
df["total_harvest_lag2"] = df["total_harvest"].shift(2)
df["harvest_change_yoy"] = df["total_harvest"] - df["total_harvest_lag1"]

# =============================
# Step 6: Formatting + type consistency (remove trailing .0)
# =============================
count_cols = [
    "gun_antlered", "gun_antlerless", "gun_total",
    "gun_licensed_hunters",
    "archery_antlered", "archery_antlerless", "archery_total",
    "archery_licensed_hunters",
    "licensed_hunters_total",
    "total_antlered", "total_antlerless", "total_harvest",
]
to_int64_safe(df, count_cols)

# =============================
# Step 7: Missing data diagnostics
# =============================
missing_counts = df.isna().sum().sort_values(ascending=False)
missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)

missing_summary = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_pct": missing_pct
}).reset_index().rename(columns={"index":"column"})

print("\nTop missing columns:")
print(missing_summary.head(15))

# Bar chart: missing counts (top 15)
top_n = 15
plot_df = missing_summary.head(top_n).iloc[::-1]  # reverse for horizontal bar readability

plt.figure(figsize=(10, 6))
plt.barh(plot_df["column"], plot_df["missing_count"])
plt.title(f"Missing Values by Column (Top {top_n})")
plt.xlabel("Missing count")
plt.ylabel("Column")
save_show("missing_values_top15.png")

# =============================
# Step 8: Core visuals for Presentation 2
# =============================
# 1) Total harvest trend
plt.figure(figsize=(10,5))
plt.plot(df["year"], df["total_harvest"], linewidth=2)
plt.title("Wisconsin Deer Harvest Trend (1966–2025)")
plt.xlabel("Year")
plt.ylabel("Total Harvest")
plt.grid(True)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
save_show("harvest_trend.png")

# 2) Weapon harvest trends
plt.figure(figsize=(10,5))
plt.plot(df["year"], df["gun_total"], label="Gun harvest", linewidth=2)
plt.plot(df["year"], df["archery_total"], label="Archery harvest (bow + crossbow)", linewidth=2)
plt.title("Weapon Harvest Trends in Wisconsin (Gun vs Archery)")
plt.xlabel("Year")
plt.ylabel("Harvest count")
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
save_show("weapon_trends.png")

# 3) WSI lag vs harvest (scatter)
plt.figure(figsize=(7,5))
plt.scatter(df["wsi_lag1"], df["total_harvest"])
plt.title("Winter Severity vs Next-Year Harvest")
plt.xlabel("WSI (lag 1)")
plt.ylabel("Total harvest")
plt.grid(True)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
save_show("wsi_lag_vs_harvest.png")

# 4) Lag1 harvest vs current harvest (scatter)
plt.figure(figsize=(7,5))
plt.scatter(df["total_harvest_lag1"], df["total_harvest"])
plt.title("Previous Year Harvest vs Current Year Harvest")
plt.xlabel("Total harvest (lag 1)")
plt.ylabel("Total harvest")
plt.grid(True)
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
save_show("harvest_lag1_vs_harvest.png")

# =============================
# Step 9: Leakage guardrails (recommendations)
# =============================
"""
Target definition recommendation (for modeling):
- Target: total_harvest at year t

Avoid leakage by NOT using:
- Any t-year derived totals that directly include the target (obvious)
- Any features that are computed using total_harvest(t) (e.g., harvest_per_hunter(t), harvest_change_yoy(t) if it uses t)
Time-safe options:
- Use lagged versions instead:
    total_harvest_lag1, total_harvest_lag2
    wsi_lag1
    population_lag1 (if you decide population impacts next year)
    licensed_hunters_lag1 (once available)
"""

# If you want: add lagged population too (often time-safe)
df["post_hunt_population_lag1"] = df["post_hunt_population"].shift(1)

# =============================
# Step 10: Save modeling-ready dataset
# =============================
df.to_csv(OUT_DATASET, index=False)
print("\nSaved dataset:", OUT_DATASET)

missing_weapon_years = df.loc[df["gun_total"].isna() | df["archery_total"].isna(), "year"].tolist()
missing_hunter_years = df.loc[df["gun_licensed_hunters"].isna() | df["archery_licensed_hunters"].isna(), "year"].tolist()

print("Years missing weapon totals:", missing_weapon_years)
print("Years missing licensed hunters:", missing_hunter_years)
print("\nDone.")