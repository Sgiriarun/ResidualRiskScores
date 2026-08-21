"""Load harmonised clinical and linked-mortality data across NHANES cycles.

Cycles handled:
    C = 2003-04 |  D = 2005-06   -> ethnicity = RIDRETH1  (NO Asian category)
    G = 2011-12 |  H = 2013-14   -> ethnicity = RIDRETH3  (HAS Asian, code 6)

This lets us (a) pool cycles for more events, and (b) OBSERVE the per-ethnic
fairness pattern cycle by cycle. Two honest taxonomy quirks fall out:
  - PREDICT has no Black/Hispanic category -> European reference (all cycles).
  - NHANES 2003-06 has no Asian category at all -> Asian only exists in G/H.

Ethnicity mapping
    RIDRETH3 (G/H): 3 White | 6 Asian | 4 Black | 1,2 Hispanic   (7 dropped)
    RIDRETH1 (C/D): 3 White |  -      | 4 Black | 1,2 Hispanic   (5 dropped)

The primary accelerometer analysis uses cycles C and D. Cycles G and H support
the exploratory wrist-worn accelerometer replication.
"""

import os
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from functools import reduce

from loader import nhanes_load, map_nzdep

MORT_BASE = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/"
             "linked_mortality/")

# sfx -> (download year, mortality file, ethnicity variable)
CYCLE_INFO = {
    "C": (2003, "NHANES_2003_2004_MORT_2019_PUBLIC.dat", "RIDRETH1"),
    "D": (2005, "NHANES_2005_2006_MORT_2019_PUBLIC.dat", "RIDRETH1"),
    "G": (2011, "NHANES_2011_2012_MORT_2019_PUBLIC.dat", "RIDRETH3"),
    "H": (2013, "NHANES_2013_2014_MORT_2019_PUBLIC.dat", "RIDRETH3"),
}
CYCLE_NAME = {"C": "2003-04", "D": "2005-06", "G": "2011-12", "H": "2013-14"}

_RETH3 = {3.0: "White", 6.0: "Asian", 4.0: "Black", 1.0: "Hispanic", 2.0: "Hispanic"}
_RETH1 = {3.0: "White", 4.0: "Black", 1.0: "Hispanic", 2.0: "Hispanic"}


def _take(df, cols):
    return df[[c for c in cols if c in df.columns]].copy()


def _load_lipids(year, sfx):
    """Cycle C: L13_C (LBXHDD). Others: TCHOL_x + HDL_x (LBDHDD)."""
    if sfx == "C":
        l = nhanes_load("L13_C.xpt", year, verbose=False)
        out = pd.DataFrame({"SEQN": l["SEQN"], "tchdl": l["LBXTC"] / l["LBXHDD"]})
    else:
        tc = nhanes_load(f"TCHOL_{sfx}.xpt", year, verbose=False)[["SEQN", "LBXTC"]]
        hd = nhanes_load(f"HDL_{sfx}.xpt", year, verbose=False)[["SEQN", "LBDHDD"]]
        out = tc.merge(hd, on="SEQN", how="inner")
        out["tchdl"] = out["LBXTC"] / out["LBDHDD"]
    return out[["SEQN", "tchdl"]]


def _load_mort(mort_file, cvd_codes):
    tmp = f"/tmp/{mort_file}"
    urllib.request.urlretrieve(MORT_BASE + mort_file, tmp)
    m = pd.read_fwf(
        tmp, colspecs=[(0, 6), (14, 15), (15, 16), (16, 19), (42, 45), (45, 48)],
        names=["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_INT", "PERMTH_EXM"],
        dtype={"UCOD_LEADING": str}, na_values=["."])
    os.remove(tmp)
    m["cvd_death"] = ((m["MORTSTAT"] == 1) &
                      (m["UCOD_LEADING"].isin(list(cvd_codes)))).astype(int)
    m["PERMTH_EXM"] = pd.to_numeric(m["PERMTH_EXM"], errors="coerce")
    m["time_years"] = m["PERMTH_EXM"] / 12.0
    return m[["SEQN", "cvd_death", "time_years"]]


def _code_suffix(cvd_codes):
    """Stable cache suffix so different endpoint definitions cannot collide."""
    codes = tuple(str(code) for code in cvd_codes)
    return "_".join(codes)


def _clinical(year, sfx, reth_var, verbose):
    if verbose:
        print(f"  [cycle {sfx} / {CYCLE_NAME[sfx]}] clinical ...")
    demo = _take(nhanes_load(f"DEMO_{sfx}.xpt", year, verbose=False),
                 ["SEQN", "RIDAGEYR", "RIAGENDR", reth_var, "INDFMPIR"])
    bp = nhanes_load(f"BPX_{sfx}.xpt", year, verbose=False)
    bp["sbp"] = bp[["BPXSY2", "BPXSY3"]].mean(axis=1)
    bp = _take(bp, ["SEQN", "sbp"])
    lip = _load_lipids(year, sfx)
    bmi = nhanes_load(f"BMX_{sfx}.xpt", year, verbose=False).rename(columns={"BMXBMI": "bmi"})
    bmi = _take(bmi, ["SEQN", "bmi"])
    diq = nhanes_load(f"DIQ_{sfx}.xpt", year, verbose=False)
    diq["diabetes"] = (diq["DIQ010"] == 1).astype(int)
    diq = _take(diq, ["SEQN", "diabetes"])
    mcq = nhanes_load(f"MCQ_{sfx}.xpt", year, verbose=False)
    pri = [c for c in ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"] if c in mcq.columns]
    mcq["prior_cvd"] = (mcq[pri] == 1).any(axis=1).astype(int)
    mcq["familyhx"] = (mcq["MCQ300A"] == 1).astype(int) if "MCQ300A" in mcq.columns else 0
    mcq = _take(mcq, ["SEQN", "prior_cvd", "familyhx"])
    bpq = nhanes_load(f"BPQ_{sfx}.xpt", year, verbose=False)
    bpq["bpl"] = (bpq["BPQ050A"] == 1).astype(int) if "BPQ050A" in bpq.columns else 0
    bpq["lld"] = (bpq["BPQ090D"] == 1).astype(int) if "BPQ090D" in bpq.columns else 0
    bpq = _take(bpq, ["SEQN", "bpl", "lld"])
    smq = nhanes_load(f"SMQ_{sfx}.xpt", year, verbose=False)
    smq["cur_smoker"] = smq["SMQ040"].isin([1, 2]).astype(int)
    smq["ex_smoker"] = (smq["SMQ020"].eq(1) & smq["SMQ040"].eq(3)).astype(int)
    smq.loc[smq["cur_smoker"] == 1, "ex_smoker"] = 0
    smq = _take(smq, ["SEQN", "cur_smoker", "ex_smoker"])
    df = reduce(lambda a, b: a.merge(b, on="SEQN", how="left"),
                [demo, bp, lip, bmi, diq, mcq, bpq, smq])
    df["cycle"] = sfx
    df["reth_var"] = reth_var
    return df


def load_equity_cycle(sfx, cvd_codes=("001", "005"), age_min=30, age_max=79,
                      use_cache=True, verbose=True):
    """Load + clean ONE cycle, all ethnic groups mapped. Cached per cycle."""
    year, mort_file, reth_var = CYCLE_INFO[sfx]
    cache_dir = Path(__file__).resolve().parent.parent / "analysis" / "data" / "source"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"equity_cycle_{sfx}_codes{_code_suffix(cvd_codes)}.parquet"
    if use_cache and cache.exists():
        if verbose:
            print(f"  cached cycle {sfx}: {cache}")
        return pd.read_parquet(cache)

    clin = _clinical(year, sfx, reth_var, verbose)
    mort = _load_mort(mort_file, cvd_codes)
    df = clin.merge(mort, on="SEQN", how="inner")

    # ── Map ethnicity using the cycle's scheme ───────────────────────────────
    mapping = _RETH3 if reth_var == "RIDRETH3" else _RETH1
    df["ethnicity"] = df[reth_var].map(mapping)
    df = df[df["ethnicity"].notna()].copy()            # drop "Other"/unmapped
    df["asian"] = (df["ethnicity"] == "Asian").astype(int)
    for c in ["maori", "pacific", "indian", "af", "athrombi"]:
        df[c] = 0
    df["predict_eth_used"] = np.where(df["asian"] == 1, "Asian", "European (reference)")
    df["has_predict_cat"] = df["ethnicity"].isin(["White", "Asian"]).astype(int)

    df["nzdep"] = map_nzdep(df["INDFMPIR"])
    df["sex"] = np.where(df["RIAGENDR"] == 2, "Female", "Male")
    df["age"] = df["RIDAGEYR"].astype(float)
    for col in ["cur_smoker", "ex_smoker", "diabetes", "bpl", "lld", "familyhx"]:
        df[col] = df[col].fillna(0).astype(int)

    df = df[(df["age"] >= age_min) & (df["age"] <= age_max)].copy()
    df = df[df["prior_cvd"] == 0].copy()
    df = df[df["time_years"].notna() & (df["time_years"] > 0)].copy()
    df = df.dropna(subset=["sbp", "tchdl"]).copy()

    final = ["SEQN", "cycle", "sex", "age", "ethnicity", "predict_eth_used",
             "has_predict_cat", "nzdep", "asian", "maori", "pacific", "indian",
             "cur_smoker", "ex_smoker", "diabetes", "af", "familyhx",
             "bpl", "lld", "athrombi", "sbp", "tchdl", "bmi",
             "cvd_death", "time_years"]
    df = df[[c for c in final if c in df.columns]].reset_index(drop=True)
    df.to_parquet(cache, index=False)
    if verbose:
        print(f"    cycle {sfx}: N={len(df)}  deaths={int(df.cvd_death.sum())}")
    return df


def load_equity_cycles(cycles=("C", "D", "G", "H"), cvd_codes=("001", "005"),
                       age_max=79, use_cache=True, verbose=True):
    """Load several cycles and concatenate (keeps the `cycle` column)."""
    if verbose:
        print("=" * 56)
        print(f"NHANES EQUITY multi-cycle: {list(cycles)}  codes={cvd_codes}")
        print("=" * 56)
    parts = [load_equity_cycle(s, cvd_codes=cvd_codes, age_max=age_max, use_cache=use_cache, verbose=verbose)
             for s in cycles]
    df = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"\nPooled N={len(df)}  total deaths={int(df.cvd_death.sum())}")
    return df


if __name__ == "__main__":
    load_equity_cycles(verbose=True)
