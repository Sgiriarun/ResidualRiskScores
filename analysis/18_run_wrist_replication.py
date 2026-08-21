"""Exploratory cross-device replication using NHANES 2011-2014 wrist MIMS.

This analysis does not modify or replace the frozen C+D primary study. It asks
only whether a single standardised activity-volume variable has a directionally
consistent association in later wrist-worn NHANES cycles. The low event count
precludes learned representations and clinical-utility claims.
"""

from __future__ import annotations

import math
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

from models import CACHE, DATA, FIGURES, HERE, PREDICTIONS, TABLES, c_index, load_config, stratified_folds

sys.path.insert(0, str(HERE.parent / "support"))
from loader_equity_multicycle import load_equity_cycle  # noqa: E402
from predict_lp import add_predict_columns  # noqa: E402


WRIST_CYCLES = {
    "G": {
        "year": 2011,
        "mortality": "NHANES_2011_2012_MORT_2019_PUBLIC.dat",
        "activity": "PAXDAY_G.xpt",
    },
    "H": {
        "year": 2013,
        "mortality": "NHANES_2013_2014_MORT_2019_PUBLIC.dat",
        "activity": "PAXDAY_H.xpt",
    },
}
MORTALITY_BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"
NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"


def ensure_file(path: Path, url: str) -> Path:
    if not path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_mortality(path: Path, endpoint_codes: tuple[str, ...]) -> pd.DataFrame:
    mortality = pd.read_fwf(
        path,
        colspecs=[(0, 6), (14, 15), (15, 16), (16, 19), (42, 45), (45, 48)],
        names=["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_INT", "PERMTH_EXM"],
        dtype={"UCOD_LEADING": str},
        na_values=["."],
    )
    mortality["cvd_death"] = (
        mortality.MORTSTAT.eq(1) & mortality.UCOD_LEADING.isin(endpoint_codes)
    ).astype(int)
    mortality["time_years"] = pd.to_numeric(mortality.PERMTH_EXM, errors="coerce") / 12.0
    return mortality[["SEQN", "cvd_death", "time_years"]]


def load_activity(path: Path) -> pd.DataFrame:
    activity = pd.read_sas(path, format="xport", encoding="utf-8")
    valid = activity[
        activity.PAXVMD.ge(600)
        & activity.PAXMTSD.notna()
        & activity.PAXMTSD.ge(0)
    ].copy()
    summary = (
        valid.groupby("SEQN")
        .agg(wrist_mims=("PAXMTSD", "mean"), valid_days=("PAXVMD", "count"))
        .reset_index()
    )
    return summary[summary.valid_days.ge(4)].copy()


def build_wrist_cohort(config: dict) -> pd.DataFrame:
    endpoint_codes = tuple(str(code) for code in config["endpoint"]["codes"])
    parts = []
    for cycle, details in WRIST_CYCLES.items():
        clinical = load_equity_cycle(
            cycle,
            cvd_codes=endpoint_codes,
            use_cache=True,
            verbose=False,
        ).drop(columns=["cvd_death", "time_years"])
        mortality_name = str(details["mortality"])
        mortality_path = ensure_file(
            CACHE / mortality_name,
            f"{MORTALITY_BASE}/{mortality_name}",
        )
        activity_name = str(details["activity"])
        activity_path = ensure_file(
            CACHE / activity_name,
            f"{NHANES_BASE}/{details['year']}/DataFiles/{activity_name}",
        )
        mortality = load_mortality(mortality_path, endpoint_codes)
        activity = load_activity(activity_path)
        merged = (
            clinical.merge(mortality, on="SEQN", how="inner", validate="one_to_one")
            .merge(activity, on="SEQN", how="inner", validate="one_to_one")
        )
        merged["cycle"] = cycle
        parts.append(merged)
    cohort = pd.concat(parts, ignore_index=True)
    cohort = cohort[cohort.time_years.notna() & cohort.time_years.gt(0)].copy()
    cohort = add_predict_columns(cohort)
    cohort = cohort[
        np.isfinite(cohort.lp_predict)
        & np.isfinite(cohort.wrist_mims)
    ].sort_values(["cycle", "SEQN"]).reset_index(drop=True)
    cohort["endpoint"] = str(config["endpoint"]["name"])
    cohort.to_parquet(DATA / "wrist_replication_cohort.parquet", index=False)
    return cohort


def cycle_standardize(train: pd.DataFrame, test: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    train_z = np.empty(len(train), float)
    test_z = np.empty(len(test), float)
    for cycle in sorted(train.cycle.unique()):
        train_mask = train.cycle.eq(cycle).to_numpy()
        test_mask = test.cycle.eq(cycle).to_numpy()
        values = train.loc[train_mask, column].to_numpy(float)
        mean = float(np.mean(values))
        scale = float(np.std(values, ddof=0))
        if not np.isfinite(scale) or scale < 1e-8:
            raise ValueError(f"Invalid {column} scale in cycle {cycle}")
        train_z[train_mask] = (values - mean) / scale
        test_z[test_mask] = (test.loc[test_mask, column].to_numpy(float) - mean) / scale
    return train_z, test_z


def fit_common_activity_coefficient(data: pd.DataFrame, activity_z: np.ndarray):
    strata = data.cycle.map({cycle: number for number, cycle in enumerate(sorted(data.cycle.unique()))})
    return PHReg(
        endog=data.time_years.to_numpy(float),
        exog=np.asarray(activity_z, float)[:, None],
        status=data.cvd_death.to_numpy(int),
        offset=data.lp_predict.to_numpy(float),
        strata=strata.to_numpy(int),
    ).fit(disp=False)


def out_of_fold_predictions(cohort: pd.DataFrame, config: dict) -> pd.DataFrame:
    folds = stratified_folds(cohort, int(config["outer_folds"]), int(config["seed"]) + 1800)
    rows = []
    all_indices = np.arange(len(cohort))
    for fold_number, test_indices in enumerate(folds, start=1):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        train = cohort.iloc[train_indices].reset_index(drop=True)
        test = cohort.iloc[test_indices].reset_index(drop=True)
        train_z, test_z = cycle_standardize(train, test, "wrist_mims")
        result = fit_common_activity_coefficient(train, train_z)
        beta = float(result.params[0])
        for local_index, participant in test.iterrows():
            rows.append(
                {
                    "SEQN": int(participant.SEQN),
                    "cycle": str(participant.cycle),
                    "outer_fold": fold_number,
                    "time_years": float(participant.time_years),
                    "cvd_death": int(participant.cvd_death),
                    "endpoint": str(participant.endpoint),
                    "wrist_mims": float(participant.wrist_mims),
                    "activity_z": float(test_z[local_index]),
                    "fold_beta": beta,
                    "predict_lp": float(participant.lp_predict),
                    "predict_wrist_lp": float(participant.lp_predict + beta * test_z[local_index]),
                }
            )
    predictions = pd.DataFrame(rows).sort_values(["cycle", "SEQN"]).reset_index(drop=True)
    if len(predictions) != len(cohort) or predictions.SEQN.duplicated().any():
        raise AssertionError("Wrist replication predictions are incomplete")
    predictions.to_parquet(PREDICTIONS / "wrist_mims_replication.parquet", index=False)
    return predictions


def paired_delta(data: pd.DataFrame) -> tuple[float, float, float]:
    base = c_index(data, data.predict_lp.to_numpy(float))
    augmented = c_index(data, data.predict_wrist_lp.to_numpy(float))
    return base, augmented, augmented - base


def bootstrap_discrimination(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    replicates = int(config["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["seed"]) + 1801)
    cycles = sorted(predictions.cycle.unique())
    point = {cycle: paired_delta(predictions[predictions.cycle.eq(cycle)].reset_index(drop=True)) for cycle in cycles}
    event_weights = {
        cycle: int(predictions.loc[predictions.cycle.eq(cycle), "cvd_death"].sum())
        for cycle in cycles
    }
    total_events = sum(event_weights.values())
    weighted_point = tuple(
        sum(event_weights[cycle] * point[cycle][position] for cycle in cycles) / total_events
        for position in range(3)
    )
    distributions = {cycle: [] for cycle in cycles}
    distributions["event_weighted"] = []
    for _ in range(replicates):
        sampled_deltas = {}
        for cycle in cycles:
            values = predictions[predictions.cycle.eq(cycle)].reset_index(drop=True)
            indices = rng.integers(0, len(values), len(values))
            sample = values.iloc[indices].reset_index(drop=True)
            try:
                sampled_deltas[cycle] = paired_delta(sample)[2]
                distributions[cycle].append(sampled_deltas[cycle])
            except Exception:
                sampled_deltas[cycle] = np.nan
        if all(np.isfinite(sampled_deltas[cycle]) for cycle in cycles):
            distributions["event_weighted"].append(
                sum(event_weights[cycle] * sampled_deltas[cycle] for cycle in cycles) / total_events
            )
    rows = []
    for cycle in cycles:
        values = predictions[predictions.cycle.eq(cycle)]
        low, high = np.percentile(distributions[cycle], [2.5, 97.5])
        rows.append(
            {
                "summary": f"cycle_{cycle}",
                "participants": len(values),
                "events": int(values.cvd_death.sum()),
                "predict_c_index": point[cycle][0],
                "predict_wrist_c_index": point[cycle][1],
                "delta_c_index": point[cycle][2],
                "ci_low": low,
                "ci_high": high,
            }
        )
    low, high = np.percentile(distributions["event_weighted"], [2.5, 97.5])
    rows.append(
        {
            "summary": "event_weighted_GH",
            "participants": len(predictions),
            "events": int(predictions.cvd_death.sum()),
            "predict_c_index": weighted_point[0],
            "predict_wrist_c_index": weighted_point[1],
            "delta_c_index": weighted_point[2],
            "ci_low": low,
            "ci_high": high,
        }
    )
    output = pd.DataFrame(rows)
    output.to_csv(TABLES / "wrist_replication_discrimination.csv", index=False)
    return output


def association(data: pd.DataFrame, activity_column: str, label: str, device: str) -> dict:
    z = np.empty(len(data), float)
    for cycle in sorted(data.cycle.unique()):
        mask = data.cycle.eq(cycle).to_numpy()
        values = data.loc[mask, activity_column].to_numpy(float)
        z[mask] = (values - values.mean()) / values.std(ddof=0)
    result = fit_common_activity_coefficient(data, z)
    beta = float(result.params[0])
    interval = np.asarray(result.conf_int())[0]
    return {
        "dataset": label,
        "device": device,
        "participants": len(data),
        "events": int(data.cvd_death.sum()),
        "activity_measure": activity_column,
        "beta_per_sd": beta,
        "hr_per_sd": math.exp(beta),
        "hr_ci_low": math.exp(float(interval[0])),
        "hr_ci_high": math.exp(float(interval[1])),
        "p_value": float(result.pvalues[0]),
    }


def build_comparison_table(wrist: pd.DataFrame, discrimination: pd.DataFrame) -> pd.DataFrame:
    hip = pd.read_parquet(DATA / "cohort.parquet")
    primary = pd.read_csv(TABLES / "primary_results.csv").set_index("model_id")
    hip_association = association(hip, "volume_mean_cpm", "NHANES 2003-2006", "Hip counts")
    wrist_association = association(wrist, "wrist_mims", "NHANES 2011-2014", "Wrist MIMS")
    wrist_summary = discrimination.set_index("summary").loc["event_weighted_GH"]
    hip_association.update(
        {
            "predict_c_index": primary.loc["M0", "c_index"],
            "activity_c_index": primary.loc["M1", "c_index"],
            "delta_c_index": primary.loc["M1", "delta_vs_M0"],
            "delta_ci_low": primary.loc["M1", "delta_vs_M0_low"],
            "delta_ci_high": primary.loc["M1", "delta_vs_M0_high"],
            "analysis_role": "Primary-cohort simple activity comparator",
        }
    )
    wrist_association.update(
        {
            "predict_c_index": wrist_summary.predict_c_index,
            "activity_c_index": wrist_summary.predict_wrist_c_index,
            "delta_c_index": wrist_summary.delta_c_index,
            "delta_ci_low": wrist_summary.ci_low,
            "delta_ci_high": wrist_summary.ci_high,
            "analysis_role": "Exploratory cross-device replication",
        }
    )
    output = pd.DataFrame([hip_association, wrist_association])
    output.to_csv(TABLES / "wrist_cross_device_replication.csv", index=False)
    return output


def make_figure(comparison: pd.DataFrame) -> None:
    labels = [
        f"{row.dataset}\n{row.device}  |  N={int(row.participants):,}, events={int(row.events)}"
        for row in comparison.itertuples()
    ]
    estimates = comparison.hr_per_sd.to_numpy(float)
    low = comparison.hr_ci_low.to_numpy(float)
    high = comparison.hr_ci_high.to_numpy(float)
    y = np.arange(len(comparison))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    colors = ["#7B8794", "#009E73"]
    for index in range(len(comparison)):
        ax.errorbar(
            estimates[index],
            y[index],
            xerr=[[estimates[index] - low[index]], [high[index] - estimates[index]]],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=4,
            markersize=7,
            linewidth=2,
        )
        ax.text(
            high[index] * 1.015,
            y[index],
            f"HR {estimates[index]:.2f} ({low[index]:.2f}-{high[index]:.2f})",
            va="center",
            fontsize=10,
        )
    ax.axvline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Hazard ratio per 1 SD higher activity (HR < 1 indicates an inverse association)")
    ax.set_title(
        "The inverse activity association is directionally consistent across devices",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    right = max(1.05, float(high.max()) * 1.32)
    left = max(0.1, float(low.min()) * 0.88)
    ax.set_xlim(left, right)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig15_wrist_cross_device_replication.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig15_wrist_cross_device_replication.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    comparison: pd.DataFrame,
    discrimination: pd.DataFrame,
    mapping_sensitivity: pd.DataFrame,
) -> None:
    wrist = comparison.iloc[1]
    direction = "inverse" if wrist.hr_per_sd < 1 else "positive"
    precision = (
        "excluded the null"
        if wrist.hr_ci_high < 1 or wrist.hr_ci_low > 1
        else "included the null"
    )
    text = f"""# Exploratory Wrist-MIMS Replication

This supplementary analysis uses the same fatal heart-or-cerebrovascular
mortality proxy as the primary paper but a later NHANES cohort and a different
device placement. It is not external validation of the hip representation.

The wrist cohort contained **{int(wrist.participants):,} participants** and
**{int(wrist.events)} events**. Per 1 SD higher mean daily wrist MIMS, the
PREDICT-offset hazard ratio was **{wrist.hr_per_sd:.2f}** (95% CI
{wrist.hr_ci_low:.2f} to {wrist.hr_ci_high:.2f}; p={wrist.p_value:.3f}). The
direction was {direction}, and the interval {precision}.

Restricting the cohort to White and Asian participants with directly available
PREDICT ethnicity categories gave HR
**{mapping_sensitivity.iloc[0].hr_per_sd:.2f}** (95% CI
{mapping_sensitivity.iloc[0].hr_ci_low:.2f} to
{mapping_sensitivity.iloc[0].hr_ci_high:.2f}) across
**{int(mapping_sensitivity.iloc[0].participants):,} participants** and
**{int(mapping_sensitivity.iloc[0].events)} events**.

The event-weighted out-of-fold C-index change was
**{wrist.delta_c_index:+.4f}** (95% CI {wrist.delta_ci_low:+.4f} to
{wrist.delta_ci_high:+.4f}). With this event count, the analysis is interpreted
only as a directional cross-device check. It does not establish incremental
discrimination, calibration, clinical utility or transport of the learned hip
representation.

Cycle-specific results are stored in
`tables/wrist_replication_discrimination.csv`.
"""
    (HERE / "WRIST_REPLICATION_RESULTS.md").write_text(text)


def main() -> None:
    config = load_config()
    cohort = build_wrist_cohort(config)
    predictions = out_of_fold_predictions(cohort, config)
    discrimination = bootstrap_discrimination(predictions, config)
    comparison = build_comparison_table(cohort, discrimination)
    direct_mapping = cohort[cohort.has_predict_cat.eq(1)].reset_index(drop=True)
    mapping_sensitivity = pd.DataFrame(
        [
            association(
                direct_mapping,
                "wrist_mims",
                "NHANES 2011-2014 direct ethnicity mapping",
                "Wrist MIMS",
            )
        ]
    )
    mapping_sensitivity.to_csv(
        TABLES / "wrist_replication_mapping_sensitivity.csv", index=False
    )
    make_figure(comparison)
    write_summary(comparison, discrimination, mapping_sensitivity)
    print(comparison.round(5).to_string(index=False))
    print("\nCycle-specific discrimination")
    print(discrimination.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
