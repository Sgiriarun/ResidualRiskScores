"""Paired clinical, tuning-policy, and subgroup checks for the paper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import calibration_slope, ipcw_brier, km_event_probability, net_benefit
from models import DATA, PREDICTIONS, TABLES, c_index, load_config


POLICIES = {
    "conservative_one_se": PREDICTIONS / "oof_predictions.parquet",
    "maximum_inner_mean": PREDICTIONS / "oof_predictions_max_mean.parquet",
}


def load_policy(path):
    predictions = pd.read_parquet(path)
    participants = (
        predictions[["SEQN", "cycle", "outer_fold", "time_years", "cvd_death"]]
        .drop_duplicates("SEQN")
        .sort_values("SEQN")
        .reset_index(drop=True)
    )
    lp = predictions.pivot(index="SEQN", columns="model_id", values="linear_predictor").loc[participants.SEQN]
    risk = predictions.pivot(index="SEQN", columns="model_id", values="risk_10y").loc[participants.SEQN]
    return participants, lp, risk


def percentile(values):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    return tuple(np.percentile(finite, [2.5, 97.5])) if len(finite) else (np.nan, np.nan)


def clinical_values(data, lp, risk, horizon):
    slope = calibration_slope(data, lp)
    observed = km_event_probability(data, horizon)
    return {
        "c_index": c_index(data, lp),
        "calibration_slope_distance": abs(slope - 1.0),
        "calibration_in_large_absolute": abs(float(np.mean(risk)) - observed),
        "brier_10y": ipcw_brier(data, risk, horizon),
        "net_benefit_5pct": net_benefit(data, risk, 0.05, horizon),
        "net_benefit_10pct": net_benefit(data, risk, 0.10, horizon),
        "net_benefit_15pct": net_benefit(data, risk, 0.15, horizon),
    }


def paired_clinical(config):
    rows = []
    horizon = float(config["horizon_years"])
    for position, (policy, path) in enumerate(POLICIES.items()):
        participants, lp, risk = load_policy(path)
        point_model = clinical_values(
            participants, lp["M6"].to_numpy(float), risk["M6"].to_numpy(float), horizon
        )
        point_reference = clinical_values(
            participants, lp["M3"].to_numpy(float), risk["M3"].to_numpy(float), horizon
        )
        distributions = {metric: [] for metric in point_model}
        rng = np.random.default_rng(int(config["seed"]) + 1001 + position)
        for _ in range(int(config["bootstrap_replicates"])):
            indices = rng.integers(0, len(participants), len(participants))
            sample = participants.iloc[indices].reset_index(drop=True)
            model = clinical_values(
                sample,
                lp["M6"].to_numpy(float)[indices],
                risk["M6"].to_numpy(float)[indices],
                horizon,
            )
            reference = clinical_values(
                sample,
                lp["M3"].to_numpy(float)[indices],
                risk["M3"].to_numpy(float)[indices],
                horizon,
            )
            for metric in distributions:
                distributions[metric].append(model[metric] - reference[metric])
        for metric, values in distributions.items():
            low, high = percentile(values)
            direction = "higher_is_better" if metric.startswith("net_benefit") or metric == "c_index" else "lower_is_better"
            rows.append(
                {
                    "policy": policy,
                    "comparison": "M6_minus_M3",
                    "metric": metric,
                    "direction": direction,
                    "M6": point_model[metric],
                    "M3": point_reference[metric],
                    "paired_difference": point_model[metric] - point_reference[metric],
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def tuning_policy_comparison(config):
    participants, conservative_lp, _ = load_policy(POLICIES["conservative_one_se"])
    other_participants, maximum_lp, _ = load_policy(POLICIES["maximum_inner_mean"])
    if not np.array_equal(participants.SEQN, other_participants.SEQN):
        raise ValueError("Tuning policies do not contain the same participants")
    rng = np.random.default_rng(int(config["seed"]) + 1003)
    bootstrap_indices = [
        rng.integers(0, len(participants), len(participants))
        for _ in range(int(config["bootstrap_replicates"]))
    ]
    rows = []
    for model_id in ("M4", "M5", "M6"):
        conservative = conservative_lp[model_id].to_numpy(float)
        maximum = maximum_lp[model_id].to_numpy(float)
        differences = []
        for indices in bootstrap_indices:
            sample = participants.iloc[indices].reset_index(drop=True)
            differences.append(c_index(sample, maximum[indices]) - c_index(sample, conservative[indices]))
        low, high = percentile(differences)
        rows.append(
            {
                "model_id": model_id,
                "conservative_one_se_c_index": c_index(participants, conservative),
                "maximum_inner_mean_c_index": c_index(participants, maximum),
                "maximum_minus_conservative": c_index(participants, maximum) - c_index(participants, conservative),
                "ci_low": low,
                "ci_high": high,
                "purpose": "diagnostic comparison of tuning policies; not a model efficacy claim",
            }
        )
    return pd.DataFrame(rows)


def subgroup_performance(config):
    cohort = pd.read_parquet(DATA / "cohort.parquet").sort_values("SEQN").reset_index(drop=True)
    groups = {
        "cycle": cohort.cycle.astype(str),
        "sex": cohort.sex.astype(str),
        "age_group": pd.cut(cohort.age, [29, 54, 79], labels=["30-54", "55-79"]).astype(str),
        "ethnicity": cohort.ethnicity.astype(str),
    }
    rows = []
    for policy_position, (policy, path) in enumerate(POLICIES.items()):
        participants, lp, _ = load_policy(path)
        if not np.array_equal(cohort.SEQN, participants.SEQN):
            raise ValueError("Cohort and prediction order differ")
        for group_name, labels in groups.items():
            for label in sorted(labels.unique()):
                mask = labels.to_numpy() == label
                subset = participants.loc[mask].reset_index(drop=True)
                events = int(subset.cvd_death.sum())
                m3 = lp.loc[mask, "M3"].to_numpy(float)
                m6 = lp.loc[mask, "M6"].to_numpy(float)
                if events < 5:
                    continue
                rng = np.random.default_rng(
                    int(config["seed"]) + 1100 + policy_position * 100 + len(rows)
                )
                differences = []
                for _ in range(int(config["bootstrap_replicates"])):
                    indices = rng.integers(0, len(subset), len(subset))
                    sample = subset.iloc[indices].reset_index(drop=True)
                    try:
                        differences.append(
                            c_index(sample, m6[indices]) - c_index(sample, m3[indices])
                        )
                    except ZeroDivisionError:
                        continue
                low, high = percentile(differences)
                rows.append(
                    {
                        "policy": policy,
                        "group_variable": group_name,
                        "group": label,
                        "participants": int(mask.sum()),
                        "events": events,
                        "M3_c_index": c_index(subset, m3),
                        "M6_c_index": c_index(subset, m6),
                        "M6_minus_M3": c_index(subset, m6) - c_index(subset, m3),
                        "ci_low": low,
                        "ci_high": high,
                        "interpretation": "descriptive subgroup estimate; not a fairness guarantee",
                    }
                )
    return pd.DataFrame(rows)


def main():
    config = load_config()
    clinical = paired_clinical(config)
    tuning = tuning_policy_comparison(config)
    subgroup = subgroup_performance(config)
    clinical.to_csv(TABLES / "paired_primary_clinical_comparison.csv", index=False)
    tuning.to_csv(TABLES / "tuning_policy_comparison.csv", index=False)
    subgroup.to_csv(TABLES / "subgroup_performance.csv", index=False)
    print("Paired primary clinical comparison")
    print(clinical.round(6).to_string(index=False))
    print("\nTuning policy comparison")
    print(tuning.round(6).to_string(index=False))
    print("\nSubgroup rows:", len(subgroup))


if __name__ == "__main__":
    main()
