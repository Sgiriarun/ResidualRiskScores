"""Recompute absolute-risk metrics at a 5-year horizon without retraining models.

The primary paper reports 10-year NHANES mortality-risk checks. Because the
published PREDICT model is a 5-year risk score, this script recomputes
fold-specific baseline hazards and absolute-risk metrics at 5 years from the
same out-of-fold linear predictors and saved train/test fold caches.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from metrics import (
    bootstrap_clinical_metrics,
    calibration_table,
    decision_curve,
    ipcw_brier,
    km_event_probability,
    net_benefit,
)
from models import (
    CACHE,
    DATA,
    MODEL_BY_ID,
    MODEL_IDS,
    PREDICTIONS,
    TABLES,
    absolute_risk_from_training,
    c_index,
    fit_head,
    load_config,
)


LEARNED_MODELS = {"M4", "M5", "M6"}


def output_path(policy: str):
    if policy == "conservative_one_se":
        return PREDICTIONS / "oof_predictions_5y.parquet"
    if policy == "maximum_inner_mean":
        return PREDICTIONS / "oof_predictions_max_mean_5y.parquet"
    raise ValueError(policy)


def source_path(policy: str):
    if policy == "conservative_one_se":
        return PREDICTIONS / "oof_predictions.parquet"
    if policy == "maximum_inner_mean":
        return PREDICTIONS / "oof_predictions_max_mean.parquet"
    raise ValueError(policy)


def table_name(stem: str, policy: str) -> str:
    suffix = "" if policy == "conservative_one_se" else "_max_mean"
    return f"horizon_5y_{stem}{suffix}.csv"


def load_fold_values(data: pd.DataFrame) -> np.ndarray:
    assignments = pd.read_csv(TABLES / "fold_assignments.csv").set_index("SEQN")
    return assignments.loc[data.SEQN.astype(int), "outer_fold"].to_numpy(int)


def load_max_mean_selection(fold: int, model_id: str) -> tuple[int, float, float]:
    selections = pd.read_csv(TABLES / "max_mean_selections.csv")
    row = selections[(selections.outer_fold == fold) & (selections.model_id == model_id)]
    if row.empty:
        raise ValueError(f"Missing max-mean selection for fold={fold}, model={model_id}")
    record = row.iloc[0]
    return int(record.selected_dimension), float(record.selected_alpha), float(record.inner_c_index)


def conservative_train_test_lp(
    data: pd.DataFrame,
    fold: int,
    model_id: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    train_data = data.iloc[train_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    if model_id == "M0":
        return (
            train_data.lp_predict.to_numpy(float),
            test_data.lp_predict.to_numpy(float),
            0,
            np.nan,
            np.nan,
        )
    with np.load(CACHE / f"features_full_fold_{fold}_{model_id}.npz") as stored:
        alpha = float(stored["selected_alpha"][0])
        dimension = int(stored["selected_dimension"][0])
        train_features = stored["train_features"]
        test_features = stored["test_features"]
    train_lp, test_lp, _ = fit_head(
        train_data,
        test_data,
        train_features,
        test_features,
        alpha,
    )
    inner = np.nan
    return train_lp, test_lp, dimension, alpha, inner


def policy_train_test_lp(
    data: pd.DataFrame,
    fold: int,
    model_id: str,
    policy: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    if policy == "conservative_one_se" or model_id not in LEARNED_MODELS:
        return conservative_train_test_lp(data, fold, model_id, train_indices, test_indices)

    dimension, alpha, inner = load_max_mean_selection(fold, model_id)
    with np.load(CACHE / f"max_mean_fold_{fold}_{model_id}.npz") as stored:
        train_lp = stored["train_lp"]
        test_lp = stored["test_lp"]
    return train_lp, test_lp, dimension, alpha, inner


def build_5y_predictions(policy: str, horizon: float) -> pd.DataFrame:
    data = pd.read_parquet(DATA / "cohort.parquet")
    source = pd.read_parquet(source_path(policy))
    source_wide = (
        source.pivot(index="SEQN", columns="model_id", values="linear_predictor")
        .loc[data.SEQN]
    )
    fold_values = load_fold_values(data)
    rows = []
    all_indices = np.arange(len(data))
    for fold in sorted(np.unique(fold_values)):
        test_indices = np.flatnonzero(fold_values == fold)
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        train_data = data.iloc[train_indices].reset_index(drop=True)
        for model_id in MODEL_IDS:
            train_lp, test_lp, dimension, alpha, inner = policy_train_test_lp(
                data,
                int(fold),
                model_id,
                policy,
                train_indices,
                test_indices,
            )
            expected = source_wide.loc[data.SEQN.iloc[test_indices], model_id].to_numpy(float)
            if not np.allclose(test_lp, expected, atol=1e-8, rtol=1e-8):
                max_diff = float(np.max(np.abs(test_lp - expected)))
                raise AssertionError(
                    f"{policy} fold={fold} {model_id} LP does not match source; max diff={max_diff:.3g}"
                )
            risk = absolute_risk_from_training(train_data, train_lp, test_lp, horizon)
            for local, global_index in enumerate(test_indices):
                participant = data.iloc[global_index]
                rows.append(
                    {
                        "SEQN": int(participant.SEQN),
                        "cycle": str(participant.cycle),
                        "outer_fold": int(fold),
                        "time_years": float(participant.time_years),
                        "cvd_death": int(participant.cvd_death),
                        "endpoint": str(participant.endpoint),
                        "model_id": model_id,
                        "model_label": MODEL_BY_ID[model_id]["label"],
                        "linear_predictor": float(test_lp[local]),
                        "risk_5y": float(risk[local]),
                        "selected_dimension": int(dimension),
                        "selected_alpha": float(alpha),
                        "inner_c_index": float(inner),
                        "horizon_years": horizon,
                        "tuning_policy": policy,
                    }
                )
    predictions = pd.DataFrame(rows).sort_values(["SEQN", "model_id"]).reset_index(drop=True)
    if len(predictions) != len(data) * len(MODEL_IDS):
        raise AssertionError(f"{policy} 5-year prediction table is incomplete")
    predictions.to_parquet(output_path(policy), index=False)
    return predictions


def rename_horizon_columns(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(
        columns={
            "mean_predicted_10y_risk": "mean_predicted_5y_risk",
            "mean_predicted_10y_risk_ci_low": "mean_predicted_5y_risk_ci_low",
            "mean_predicted_10y_risk_ci_high": "mean_predicted_5y_risk_ci_high",
            "observed_10y_km_risk": "observed_5y_km_risk",
            "observed_10y_km_risk_ci_low": "observed_5y_km_risk_ci_low",
            "observed_10y_km_risk_ci_high": "observed_5y_km_risk_ci_high",
            "brier_10y": "brier_5y",
            "brier_10y_ci_low": "brier_5y_ci_low",
            "brier_10y_ci_high": "brier_5y_ci_high",
        }
    )


def clinical_values(data: pd.DataFrame, lp: np.ndarray, risk: np.ndarray, horizon: float) -> dict[str, float]:
    from metrics import calibration_slope

    slope = calibration_slope(data, lp)
    observed = km_event_probability(data, horizon)
    return {
        "c_index": c_index(data, lp),
        "calibration_slope_distance": abs(slope - 1.0),
        "calibration_in_large_absolute": abs(float(np.mean(risk)) - observed),
        "brier_5y": ipcw_brier(data, risk, horizon),
        "net_benefit_5pct": net_benefit(data, risk, 0.05, horizon),
        "net_benefit_10pct": net_benefit(data, risk, 0.10, horizon),
        "net_benefit_15pct": net_benefit(data, risk, 0.15, horizon),
    }


def percentile(values):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    return tuple(np.percentile(finite, [2.5, 97.5])) if len(finite) else (np.nan, np.nan)


def paired_primary_comparison(
    participants: pd.DataFrame,
    lp_wide: pd.DataFrame,
    risk_wide: pd.DataFrame,
    config: dict,
    policy: str,
    horizon: float,
) -> pd.DataFrame:
    model = clinical_values(
        participants,
        lp_wide["M6"].to_numpy(float),
        risk_wide["M6"].to_numpy(float),
        horizon,
    )
    reference = clinical_values(
        participants,
        lp_wide["M3"].to_numpy(float),
        risk_wide["M3"].to_numpy(float),
        horizon,
    )
    distributions = {metric: [] for metric in model}
    rng = np.random.default_rng(int(config["seed"]) + 1701 + (0 if policy == "conservative_one_se" else 100))
    for _ in range(int(config["bootstrap_replicates"])):
        indices = rng.integers(0, len(participants), len(participants))
        sample = participants.iloc[indices].reset_index(drop=True)
        sampled_model = clinical_values(
            sample,
            lp_wide["M6"].to_numpy(float)[indices],
            risk_wide["M6"].to_numpy(float)[indices],
            horizon,
        )
        sampled_reference = clinical_values(
            sample,
            lp_wide["M3"].to_numpy(float)[indices],
            risk_wide["M3"].to_numpy(float)[indices],
            horizon,
        )
        for metric in distributions:
            distributions[metric].append(sampled_model[metric] - sampled_reference[metric])
    rows = []
    for metric, values in distributions.items():
        low, high = percentile(values)
        direction = "higher_is_better" if metric == "c_index" or metric.startswith("net_benefit") else "lower_is_better"
        rows.append(
            {
                "horizon_years": horizon,
                "policy": policy,
                "comparison": "M6_minus_M3",
                "metric": metric,
                "direction": direction,
                "M6": model[metric],
                "M3": reference[metric],
                "paired_difference": model[metric] - reference[metric],
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def evaluate_policy(
    predictions: pd.DataFrame,
    config: dict,
    policy: str,
    horizon: float,
    *,
    run_model_metrics: bool = True,
    run_paired: bool = True,
) -> None:
    participants = (
        predictions[["SEQN", "cycle", "outer_fold", "time_years", "cvd_death", "endpoint"]]
        .drop_duplicates("SEQN")
        .sort_values("SEQN")
        .reset_index(drop=True)
    )
    lp_wide = predictions.pivot(index="SEQN", columns="model_id", values="linear_predictor").loc[participants.SEQN]
    risk_wide = predictions.pivot(index="SEQN", columns="model_id", values="risk_5y").loc[participants.SEQN]
    linear_predictors = {model_id: lp_wide[model_id].to_numpy(float) for model_id in MODEL_IDS}
    risks = {model_id: risk_wide[model_id].to_numpy(float) for model_id in MODEL_IDS}
    if run_model_metrics:
        clinical = bootstrap_clinical_metrics(
            participants,
            linear_predictors,
            risks,
            horizon,
            int(config["bootstrap_replicates"]),
            int(config["seed"]) + 1702 + (0 if policy == "conservative_one_se" else 100),
        )
        clinical = rename_horizon_columns(clinical)
        clinical["model_label"] = clinical.model_id.map(lambda value: MODEL_BY_ID[value]["label"])
        clinical["horizon_years"] = horizon
        clinical["tuning_policy"] = policy
        clinical.to_csv(TABLES / table_name("clinical_metrics", policy), index=False)

        dca = decision_curve(participants, risks, horizon)
        dca.insert(0, "horizon_years", horizon)
        dca.insert(1, "tuning_policy", policy)
        dca.to_csv(TABLES / table_name("decision_curve", policy), index=False)

        calibration = calibration_table(participants, risks, horizon).rename(
            columns={
                "mean_predicted_10y_risk": "mean_predicted_5y_risk",
                "observed_10y_km_risk": "observed_5y_km_risk",
            }
        )
        calibration.insert(0, "horizon_years", horizon)
        calibration.insert(1, "tuning_policy", policy)
        calibration.to_csv(TABLES / table_name("calibration_by_decile", policy), index=False)

    if run_paired:
        paired = paired_primary_comparison(participants, lp_wide, risk_wide, config, policy, horizon)
        paired.to_csv(TABLES / table_name("paired_primary_clinical_comparison", policy), index=False)
        print(f"\n5-year paired primary clinical comparison ({policy})")
        print(paired.round(6).to_string(index=False))


def write_horizon_summary(horizon: float) -> None:
    conservative_path = TABLES / table_name("paired_primary_clinical_comparison", "conservative_one_se")
    maximum_path = TABLES / table_name("paired_primary_clinical_comparison", "maximum_inner_mean")
    clinical_path = TABLES / table_name("clinical_metrics", "conservative_one_se")
    if not all(path.exists() for path in [conservative_path, maximum_path, clinical_path]):
        return
    conservative = pd.read_csv(conservative_path).set_index("metric")
    maximum = pd.read_csv(maximum_path).set_index("metric")
    clinical = pd.read_csv(clinical_path).set_index("model_id")
    observed = float(clinical.loc["M3", "observed_5y_km_risk"])
    text = f"""# Five-Year Horizon Sensitivity

Generated directly from the saved five-year out-of-fold predictions. The same
linear predictors are used at five and ten years; only fold-specific baseline
hazards and horizon-dependent absolute-risk metrics are recomputed.

## Conservative One-SE Policy

Observed five-year heart-or-stroke mortality risk was **{observed:.4%}**. M6
changed five-year IPCW Brier score by
**{conservative.loc['brier_5y', 'paired_difference']:+.5f}** versus M3 (95% CI
{conservative.loc['brier_5y', 'ci_low']:+.5f} to
{conservative.loc['brier_5y', 'ci_high']:+.5f}). Five-percent net benefit changed
by **{conservative.loc['net_benefit_5pct', 'paired_difference']:+.5f}** (95% CI
{conservative.loc['net_benefit_5pct', 'ci_low']:+.5f} to
{conservative.loc['net_benefit_5pct', 'ci_high']:+.5f}). Calibration-slope
distance changed by
**{conservative.loc['calibration_slope_distance', 'paired_difference']:+.4f}**
(95% CI {conservative.loc['calibration_slope_distance', 'ci_low']:+.4f} to
{conservative.loc['calibration_slope_distance', 'ci_high']:+.4f}); lower is
better.

## Maximum-Inner-Mean Policy

M6 changed five-year IPCW Brier score by
**{maximum.loc['brier_5y', 'paired_difference']:+.5f}** versus M3 (95% CI
{maximum.loc['brier_5y', 'ci_low']:+.5f} to
{maximum.loc['brier_5y', 'ci_high']:+.5f}). Five-percent net benefit changed by
**{maximum.loc['net_benefit_5pct', 'paired_difference']:+.5f}** (95% CI
{maximum.loc['net_benefit_5pct', 'ci_low']:+.5f} to
{maximum.loc['net_benefit_5pct', 'ci_high']:+.5f}).

## Interpretation

The five-year analysis agrees with the ten-year conclusion. M6 has a calibration
slope closer to one, but Brier-score changes are very small and uncertain, and
paired net-benefit intervals include zero. Because observed five-year mortality
risk is below 1%, the 10% and 15% analytic thresholds classify almost nobody and
do not provide useful evidence of clinical utility.
"""
    (TABLES.parent / "HORIZON_SENSITIVITY_RESULTS.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("all", "conservative_one_se", "maximum_inner_mean"), default="all")
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument(
        "--stage",
        choices=("all", "predictions", "model_metrics", "paired"),
        default="all",
        help="Run the complete analysis or resume one independently saved stage.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=None,
        help="Override the configured bootstrap count for development runs.",
    )
    args = parser.parse_args()
    config = load_config()
    if args.bootstrap_replicates is not None:
        if args.bootstrap_replicates < 1:
            raise ValueError("--bootstrap-replicates must be positive")
        config = dict(config)
        config["bootstrap_replicates"] = int(args.bootstrap_replicates)
    policies = (
        ["conservative_one_se", "maximum_inner_mean"]
        if args.policy == "all"
        else [args.policy]
    )
    for policy in policies:
        if args.stage in {"all", "predictions"}:
            predictions = build_5y_predictions(policy, float(args.horizon))
        else:
            path = output_path(policy)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}; run --stage predictions for {policy} first"
                )
            predictions = pd.read_parquet(path)
        if args.stage == "predictions":
            continue
        evaluate_policy(
            predictions,
            config,
            policy,
            float(args.horizon),
            run_model_metrics=args.stage in {"all", "model_metrics"},
            run_paired=args.stage in {"all", "paired"},
        )
    if args.stage in {"all", "model_metrics", "paired"}:
        write_horizon_summary(float(args.horizon))


if __name__ == "__main__":
    main()
