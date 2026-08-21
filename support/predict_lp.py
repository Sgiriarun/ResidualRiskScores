"""
predict_lp.py — Compute PREDICT NoPriorCVDRisk_BMI linear predictor.

Uses published NZ coefficients from Pylypchuk et al., Lancet 2018 /
HISO 10071:2019. Nothing is re-estimated from NHANES.

These are the same coefficients hardcoded in webapp/models/predict_risk.py.
"""

import math
import numpy as np
import pandas as pd


# ── Baseline survival at 5 years (from published PREDICT paper) ───────────────
S0_FEMALE = 0.9845026
S0_MALE   = 0.9712501


# ── BMI category flags (PREDICT primary model) ───────────────────────────────

def _bmi_flags(bmi):
    """Return (bl185, b2530, b3035, b3540, bge40, bmiss) tuple."""
    if bmi is None or (isinstance(bmi, float) and np.isnan(bmi)):
        return 0, 0, 0, 0, 0, 1
    return (
        int(bmi < 18.5),
        int(25.0 <= bmi < 30.0),
        int(30.0 <= bmi < 35.0),
        int(35.0 <= bmi < 40.0),
        int(bmi >= 40.0),
        0,
    )


# ── LP computation for one row ────────────────────────────────────────────────

def compute_lp(row) -> float:
    """
    Compute PREDICT NoPriorCVDRisk_BMI linear predictor for one subject.

    Parameters
    ----------
    row : dict-like (DataFrame row or dict)

    Returns
    -------
    float : linear predictor value (log-hazard scale)
    """
    sex   = row["sex"]
    age   = max(30, min(79, float(row["age"])))
    nzdep = float(row["nzdep"])

    # Ethnic flags
    asian   = int(row.get("asian",   0))
    maori   = int(row.get("maori",   0))
    pacific = int(row.get("pacific", 0))
    indian  = int(row.get("indian",  0))

    # Lifestyle
    cur_smk  = int(row.get("cur_smoker", 0))
    ex_smk   = int(row.get("ex_smoker",  0))
    if cur_smk:
        ex_smk = 0   # cannot be both

    # Conditions and medications
    diabetes = int(row.get("diabetes", 0))
    af       = int(row.get("af",       0))
    familyhx = int(row.get("familyhx", 0))
    bpl      = int(row.get("bpl",      0))
    lld      = int(row.get("lld",      0))
    athrombi = int(row.get("athrombi", 0))   # 0 for all NHANES subjects

    # Continuous labs
    sbp   = float(row["sbp"])   if pd.notna(row["sbp"])   else 128.0
    tchdl = float(row["tchdl"]) if pd.notna(row["tchdl"]) else 3.72
    bmi   = row.get("bmi", None)
    if bmi is not None and pd.isna(bmi):
        bmi = None

    bl185, b2530, b3035, b3540, bge40, bmiss = _bmi_flags(bmi)

    if sex == "Female":
        # Recentering constants (NZ population means from PREDICT paper)
        ac = age   - 56.05801
        nc = nzdep - 2.994877
        sc = sbp   - 128.6736
        tc = tchdl - 3.715383

        # Interaction terms
        iad = 0.0 if diabetes == 0 else ac     # Age × Diabetes
        ias = ac * sc                           # Age × SBP
        ibp = 0.0 if bpl == 0     else sc      # SBP × BP medication

        lp = (
            0.0734393  * ac       +
            0.4164622  * maori    +
            0.2268597  * pacific  +
            0.2086713  * indian   +
           -0.2680559  * asian    +
            0.1444243  * ex_smk   +
            0.6768396  * cur_smk  +
            0.0957229  * nc       +
            0.4967444  * diabetes +
            0.9293084  * af       +
            0.0645588  * familyhx +
           -0.0568366  * lld      +
            0.1393368  * athrombi +
            0.3487781  * bpl      +
            0.0176523  * sc       +
            0.1361335  * tc       +
            0.6277962  * bl185    +
            0.0018215  * b2530    +
           -0.0169324  * b3035    +
            0.0343351  * b3540    +
            0.3196519  * bge40    +
            0.0213595  * bmiss    +
           -0.0189779  * iad      +
           -0.000471   * ias      +
           -0.0054002  * ibp
        )
        return lp

    else:  # Male
        ac = age   - 51.59444
        nc = nzdep - 2.975732
        sc = sbp   - 128.8637
        tc = tchdl - 4.385853

        iad = 0.0 if diabetes == 0 else ac
        ias = ac * sc
        ibp = 0.0 if bpl == 0     else sc

        lp = (
            0.0669484  * ac       +
            0.3166164  * maori    +
            0.2217931  * pacific  +
            0.3666816  * indian   +
           -0.4131973  * asian    +
            0.0748648  * ex_smk   +
            0.5317607  * cur_smk  +
            0.0631146  * nc       +
            0.4107586  * diabetes +
            0.6250334  * af       +
            0.1275721  * familyhx +
           -0.0256429  * lld      +
            0.0701999  * athrombi +
            0.2847596  * bpl      +
            0.0179827  * sc       +
            0.1296756  * tc       +
            0.5488212  * bl185    +
           -0.033177   * b2530    +
           -0.0025986  * b3035    +
            0.1202739  * b3540    +
            0.3799261  * bge40    +
           -0.073928   * bmiss    +
           -0.0124356  * iad      +
           -0.0004931  * ias      +
           -0.0049226  * ibp
        )
        return lp


def _lp_to_risk(lp: float, s0: float) -> float:
    """Convert linear predictor to 5-year risk: 1 - S0^exp(LP)."""
    return 1.0 - s0 ** math.exp(lp)


# ── Apply to full DataFrame ───────────────────────────────────────────────────

def add_predict_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lp_predict and risk_predict columns to the DataFrame.

    lp_predict  : linear predictor (log-hazard scale)
    risk_predict : 5-year CVD risk (0-1)

    Uses sex-specific baseline survivals from PREDICT paper:
      Female: S0 = 0.9845026
      Male:   S0 = 0.9712501
    """
    df = df.copy()

    lp_values = df.apply(compute_lp, axis=1)
    df["lp_predict"] = lp_values

    s0_values = df["sex"].map({"Female": S0_FEMALE, "Male": S0_MALE})
    df["risk_predict"] = [
        _lp_to_risk(lp, s0)
        for lp, s0 in zip(lp_values, s0_values)
    ]

    return df
