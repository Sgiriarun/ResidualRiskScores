"""Minimal NHANES download helpers used by the release cohort builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests


def nhanes_load(filename: str, year: int, verbose: bool = True) -> pd.DataFrame:
    """Download one public NHANES XPT file and return it as a data frame."""
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{filename}"
    if verbose:
        print(f"Downloading {filename}", flush=True)
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    response.raise_for_status()
    temporary = Path("/tmp") / filename
    temporary.write_bytes(response.content)
    try:
        return pd.read_sas(temporary, format="xport", encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)


def map_nzdep(income_ratio: pd.Series) -> pd.Series:
    """Map NHANES poverty-income ratio to the five-level proxy used by PREDICT."""
    values = pd.to_numeric(income_ratio, errors="coerce")
    mapped = np.where(
        values.isna(),
        3,
        np.where(
            values <= 1.0,
            5,
            np.where(
                values <= 2.0,
                4,
                np.where(values <= 3.0, 3, np.where(values <= 4.0, 2, 1)),
            ),
        ),
    )
    return pd.Series(mapped, index=income_ratio.index, dtype=float)
