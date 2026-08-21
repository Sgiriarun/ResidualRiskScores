"""Evaluate every canonical model from the single saved OOF prediction file."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from metrics import (
    bootstrap_clinical_metrics,
    bootstrap_discrimination,
    calibration_table,
    decision_curve,
)
from models import HERE, MODEL_BY_ID, MODEL_IDS, PREDICTIONS, TABLES, load_config


def output_name(stem: str, smoke: bool) -> str:
    return f"{stem}_smoke.csv" if smoke else f"{stem}.csv"


def load_oof(smoke: bool):
    path = PREDICTIONS / ("oof_predictions_smoke.parquet" if smoke else "oof_predictions.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run 02_run_nested_models.py first")
    predictions = pd.read_parquet(path)
    expected = set(MODEL_IDS)
    if set(predictions.model_id.unique()) != expected:
        raise ValueError(f"Prediction models differ from registry: {sorted(predictions.model_id.unique())}")
    counts = predictions.groupby("SEQN").model_id.nunique()
    if not (counts == len(MODEL_IDS)).all():
        raise ValueError("Every participant must have exactly one prediction from every model")
    participant = (
        predictions[["SEQN", "cycle", "outer_fold", "time_years", "cvd_death", "endpoint"]]
        .drop_duplicates("SEQN")
        .sort_values("SEQN")
        .reset_index(drop=True)
    )

    def pivot(column: str) -> dict[str, np.ndarray]:
        wide = predictions.pivot(index="SEQN", columns="model_id", values=column).loc[participant.SEQN]
        return {model_id: wide[model_id].to_numpy(float) for model_id in MODEL_IDS}

    return predictions, participant, pivot("linear_predictor"), pivot("risk_10y")


def selection_frequency(predictions: pd.DataFrame) -> pd.DataFrame:
    selections = predictions[
        ["outer_fold", "model_id", "selected_dimension", "selected_alpha", "inner_c_index"]
    ].drop_duplicates()
    grouped = (
        selections.groupby(["model_id", "selected_dimension", "selected_alpha"], dropna=False)
        .agg(folds_selected=("outer_fold", "size"), mean_inner_c_index=("inner_c_index", "mean"))
        .reset_index()
    )
    grouped["selection_fraction"] = grouped.folds_selected / selections.outer_fold.nunique()
    return grouped


def write_summary(
    primary: pd.DataFrame,
    clinical: pd.DataFrame,
    predictions: pd.DataFrame,
    participants: pd.DataFrame,
    config: dict,
) -> None:
    primary = primary.set_index("model_id")
    clinical = clinical.set_index("model_id")
    best_id = primary.c_index.idxmax()
    ae = primary.loc["M5"]
    combined = primary.loc["M6"]
    ae_utility = clinical.loc["M5"]
    selection = (
        predictions[["outer_fold", "model_id", "selected_dimension", "selected_alpha"]]
        .drop_duplicates()
    )
    selected_ae = selection[selection.model_id == "M5"]
    endpoint_label = config.get("endpoint", {}).get("label", "CVD deaths")
    endpoint_description = config.get("endpoint", {}).get("description", "mortality endpoint")
    if ae.delta_vs_M3_low > 0:
        beyond = "The autoencoder improved discrimination beyond circadian features."
    elif ae.delta_vs_M3_high < 0:
        beyond = "The autoencoder was slightly lower than the circadian model in the paired comparison."
    else:
        beyond = "There was no clear discrimination improvement beyond circadian features."
    if combined.delta_vs_M3_low > 0:
        combined_text = "Adding the AE to circadian features produced a paired gain with a confidence interval above zero."
    else:
        combined_text = "Adding the AE to circadian features did not produce a clear paired gain."
    text = f"""# Standardized average-day study results

Generated directly from `predictions/oof_predictions.parquet`.

## Cohort

The canonical NHANES C+D cohort contained **{len(participants):,} participants** and
**{int(participants.cvd_death.sum())} {endpoint_label}**. The endpoint is a
**{endpoint_description}**, not the full non-fatal/fatal PREDICT outcome. All reported model
predictions are out of fold.

## Primary discrimination

The transported PREDICT linear predictor achieved a C-index of **{primary.loc['M0','c_index']:.4f}**
(95% CI {primary.loc['M0','ci_low']:.4f} to {primary.loc['M0','ci_high']:.4f}).
The highest point estimate was **{MODEL_BY_ID[best_id]['label']}** at
**{primary.loc[best_id,'c_index']:.4f}**. The nested autoencoder changed C-index
by **{ae.delta_vs_M0:+.4f}** versus PREDICT (paired 95% CI
{ae.delta_vs_M0_low:+.4f} to {ae.delta_vs_M0_high:+.4f}) and
**{ae.delta_vs_M3:+.4f}** versus circadian features (paired 95% CI
{ae.delta_vs_M3_low:+.4f} to {ae.delta_vs_M3_high:+.4f}). {beyond}

The circadian-plus-AE model changed C-index by **{combined.delta_vs_M0:+.4f}**
versus PREDICT and **{combined.delta_vs_M3:+.4f}** versus circadian alone, but
both paired confidence intervals included zero. {combined_text}

## Model selection

Across the outer folds, M5 selected dimensions
**{', '.join(map(str, selected_ae.selected_dimension.astype(int).tolist()))}** and
ridge penalties **{', '.join(f'{value:g}' for value in selected_ae.selected_alpha)}**.
These choices were made entirely inside the corresponding outer training sets.

## Absolute-risk performance

For M5, the 10-year IPCW Brier score was **{ae_utility.brier_10y:.5f}** and the
calibration slope was **{ae_utility.calibration_slope:.3f}**. Net benefit at the
exploratory 5% threshold was **{ae_utility.net_benefit_5pct:.5f}**, compared with
**{clinical.loc['M0','net_benefit_5pct']:.5f}** for the transported PREDICT LP. M6 had a Brier
score of **{clinical.loc['M6','brier_10y']:.5f}** and 5% net benefit of
**{clinical.loc['M6','net_benefit_5pct']:.5f}**. Thus, the higher M6 C-index point
estimate did not produce a coherent absolute-risk or decision-utility improvement.
These thresholds are analytic checks for a transported mortality endpoint, not
validated treatment cut-points.

## Interpretation rule

The paper should claim improvement only when the paired confidence interval is
above zero. A higher point estimate without that support is described as a small,
uncertain ranking change. Clinical utility is claimed only if calibration, Brier
score, and decision-curve results are directionally coherent rather than selecting
one favourable threshold after inspection.
"""
    (HERE / "RESULTS_SUMMARY.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.smoke)
    predictions, participants, linear_predictors, risks = load_oof(args.smoke)
    replicates = int(config["bootstrap_replicates"])
    horizon = float(config["horizon_years"])
    print(f"Evaluating {len(participants):,} participants with {replicates} bootstrap resamples")
    primary = bootstrap_discrimination(
        participants, linear_predictors, replicates, int(config["seed"]) + 701
    )
    primary["model_label"] = primary.model_id.map(lambda value: MODEL_BY_ID[value]["label"])
    clinical = bootstrap_clinical_metrics(
        participants,
        linear_predictors,
        risks,
        horizon,
        replicates,
        int(config["seed"]) + 702,
    )
    clinical["model_label"] = clinical.model_id.map(lambda value: MODEL_BY_ID[value]["label"])
    dca = decision_curve(participants, risks, horizon)
    calibration = calibration_table(participants, risks, horizon)
    frequency = selection_frequency(predictions)
    primary.to_csv(TABLES / output_name("primary_results", args.smoke), index=False)
    clinical.to_csv(TABLES / output_name("clinical_metrics", args.smoke), index=False)
    dca.to_csv(TABLES / output_name("decision_curve", args.smoke), index=False)
    calibration.to_csv(TABLES / output_name("calibration_by_decile", args.smoke), index=False)
    frequency.to_csv(TABLES / output_name("selection_frequency", args.smoke), index=False)
    if args.smoke:
        target = HERE / "RESULTS_SUMMARY_SMOKE.md"
        original = HERE / "RESULTS_SUMMARY.md"
        write_summary(primary, clinical, predictions, participants, config)
        target.write_text(original.read_text())
        original.write_text(
            "# Standardized average-day study results\n\n"
            "The full results have not been generated yet. Smoke-test results are in "
            "`RESULTS_SUMMARY_SMOKE.md`.\n"
        )
    else:
        write_summary(primary, clinical, predictions, participants, config)
    print(primary[["model_id", "c_index", "ci_low", "ci_high", "delta_vs_M0", "delta_vs_M3"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
