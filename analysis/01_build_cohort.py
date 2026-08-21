"""Build the canonical C+D average-day cohort and deterministic outer folds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models import (
    DATA,
    HERE,
    MINUTES,
    MODEL_REGISTRY,
    TABLES,
    circadian_features,
    cyclic_interpolate,
    load_config,
    stratified_folds,
)

sys.path.insert(0, str(HERE.parent / "support"))
from loader_equity_multicycle import load_equity_cycle  # noqa: E402
from predict_lp import add_predict_columns  # noqa: E402


def endpoint_config(config: dict) -> dict:
    return config.get(
        "endpoint",
        {
            "codes": ["001", "005"],
            "name": "heart_or_stroke_death",
            "label": "heart-or-stroke deaths",
            "description": "fatal heart-or-cerebrovascular mortality proxy",
        },
    )


def weekly_source(cycle: str) -> Path:
    path = DATA / "source" / f"weekly_{cycle}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. The saved wear masks are required so non-wear is not treated as zero."
        )
    return path


def reconstruct_cycle(cycle: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Reconstruct one average day from observed wear minutes in the weekly cache."""
    with np.load(weekly_source(cycle)) as weekly:
        seqn = weekly["seqn"].astype(int)
        counts = weekly["counts"].astype(np.float32)
        wear = weekly["wear_mask"].astype(bool)
        valid_day = weekly["valid_day_mask"].astype(bool)
        daily_mvpa = weekly["daily_mvpa"].astype(float)
    observations = wear.sum(axis=1)
    raw = np.divide(
        (counts * wear).sum(axis=1),
        observations,
        out=np.full((len(seqn), MINUTES), np.nan, dtype=np.float32),
        where=observations > 0,
    )
    observed = observations > 0
    completed = np.stack([cyclic_interpolate(row, mask) for row, mask in zip(raw, observed)]).astype(np.float32)
    total_observed = wear.sum(axis=(1, 2))
    volume = np.divide(
        (counts * wear).sum(axis=(1, 2)),
        total_observed,
        out=np.full(len(seqn), np.nan),
        where=total_observed > 0,
    )
    mvpa = np.nanmean(daily_mvpa, axis=1)
    summary = pd.DataFrame(
        {
            "SEQN": seqn,
            "cycle": cycle,
            "volume_mean_cpm": volume,
            "mvpa_min_day": mvpa,
            "valid_days": valid_day.sum(axis=1),
            "observed_clock_minutes": observed.sum(axis=1),
            "interpolated_clock_minutes": MINUTES - observed.sum(axis=1),
        }
    )

    metrics = {
        "cycle": cycle,
        "participants": len(seqn),
        "participants_with_interpolation": int((~observed).any(axis=1).sum()),
        "maximum_interpolated_clock_minutes": int((~observed).sum(axis=1).max()),
        "minimum_valid_days": int(valid_day.sum(axis=1).min()),
        "mean_valid_days": float(valid_day.sum(axis=1).mean()),
        "mean_volume_cpm": float(np.mean(volume)),
    }
    return summary, completed, observed, metrics


def cohort_characteristics(data: pd.DataFrame, endpoint_label: str) -> pd.DataFrame:
    def mean_sd(values):
        values = pd.to_numeric(values, errors="coerce").dropna()
        return f"{values.mean():.1f} ({values.std(ddof=1):.1f})"

    rows = [
        ("Participants", f"{len(data):,}"),
        (endpoint_label.capitalize(), f"{int(data.cvd_death.sum()):,} ({100 * data.cvd_death.mean():.1f}%)"),
        ("Age, years", mean_sd(data.age)),
        ("Female", f"{int((data.sex == 'Female').sum()):,} ({100 * (data.sex == 'Female').mean():.1f}%)"),
        ("Body-mass index, kg/m2", mean_sd(data.bmi)),
        ("Systolic blood pressure, mmHg", mean_sd(data.sbp)),
        ("Valid accelerometer days", mean_sd(data.valid_days)),
        ("Total volume, counts/min", mean_sd(data.volume_mean_cpm)),
        ("MVPA, min/day", mean_sd(data.mvpa_min_day)),
        ("M10, counts/min", mean_sd(data.M10)),
        ("L5, counts/min", mean_sd(data.L5)),
        ("Relative amplitude", mean_sd(data.RA)),
    ]
    return pd.DataFrame(rows, columns=["characteristic", "overall"])


def main() -> None:
    config = load_config()
    endpoint = endpoint_config(config)
    endpoint_codes = tuple(str(code) for code in endpoint["codes"])
    endpoint_name = str(endpoint["name"])
    endpoint_label = str(endpoint["label"])
    summaries, profiles, masks, validation = [], [], [], []
    for cycle in ("C", "D"):
        summary, profile, mask, metrics = reconstruct_cycle(cycle)
        summaries.append(summary)
        profiles.append(profile)
        masks.append(mask)
        validation.append(metrics)
    activity = pd.concat(summaries, ignore_index=True)
    profile = np.concatenate(profiles)
    observed = np.concatenate(masks)

    clinical = pd.concat(
        [
            load_equity_cycle(cycle, cvd_codes=endpoint_codes, use_cache=True, verbose=False)
            for cycle in ("C", "D")
        ],
        ignore_index=True,
    )
    clinical["SEQN"] = clinical.SEQN.astype(int)
    clinical = add_predict_columns(clinical)
    activity["_activity_index"] = np.arange(len(activity))
    data = activity.merge(clinical, on=["SEQN", "cycle"], how="inner", validate="one_to_one")
    if len(data) == 0:
        raise ValueError("Activity and clinical data have no overlapping participants")
    selected = data.pop("_activity_index").to_numpy(int)
    profile = profile[selected]
    observed = observed[selected]
    if not np.array_equal(data.SEQN.to_numpy(int), activity.iloc[selected].SEQN.to_numpy(int)):
        raise ValueError("Cohort merge changed participant alignment")
    circadian = circadian_features(profile)
    data[["M10", "L5", "RA"]] = circadian
    data["endpoint"] = endpoint_name

    folds = stratified_folds(data, int(config["outer_folds"]), int(config["seed"]))
    fold_number = np.full(len(data), -1, dtype=int)
    for number, indices in enumerate(folds, start=1):
        fold_number[indices] = number
    if (fold_number < 1).any():
        raise AssertionError("Some participants were not assigned to an outer fold")
    fold_table = data[["SEQN", "cycle", "cvd_death", "endpoint"]].copy()
    fold_table["outer_fold"] = fold_number

    data.to_parquet(DATA / "cohort.parquet", index=False)
    np.savez_compressed(
        DATA / "average_day_profiles.npz",
        seqn=data.SEQN.to_numpy(int),
        profile=profile,
        observed_mask=observed,
    )
    fold_table.to_csv(TABLES / "fold_assignments.csv", index=False)
    pd.DataFrame(MODEL_REGISTRY).to_csv(TABLES / "model_registry.csv", index=False)
    pd.DataFrame(validation).to_csv(TABLES / "profile_validation.csv", index=False)
    cohort_characteristics(data, endpoint_label).to_csv(TABLES / "cohort_characteristics.csv", index=False)
    flow = pd.DataFrame(
        [
            {"stage": "Valid accelerometer profiles", "participants": len(activity), "events": np.nan},
            {
                "stage": f"Clinical and mortality eligible ({endpoint_label})",
                "participants": len(clinical),
                "events": int(clinical.cvd_death.sum()),
            },
            {"stage": "Canonical merged cohort", "participants": len(data), "events": int(data.cvd_death.sum())},
        ]
    )
    flow.to_csv(TABLES / "cohort_flow.csv", index=False)
    print(f"Canonical cohort: N={len(data):,}; {endpoint_label}={int(data.cvd_death.sum())}")
    print(pd.DataFrame(validation).round(5).to_string(index=False))
    print("Outer-fold event/cycle counts:")
    print(fold_table.groupby(["outer_fold", "cycle", "cvd_death"]).size().to_string())


if __name__ == "__main__":
    main()
