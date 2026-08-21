"""Generate the paper's locked, paired discrimination contrasts from OOF predictions."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from models import MODEL_BY_ID, PREDICTIONS, TABLES, c_index, load_config


CONTRASTS = (
    {
        "contrast_id": "C1",
        "tier": "secondary",
        "model_id": "M1",
        "reference_id": "M0",
        "question": "Does total activity volume add ranking information?",
    },
    {
        "contrast_id": "C2",
        "tier": "secondary",
        "model_id": "M2",
        "reference_id": "M0",
        "question": "Does MVPA add ranking information?",
    },
    {
        "contrast_id": "C3",
        "tier": "secondary",
        "model_id": "M3",
        "reference_id": "M0",
        "question": "Do interpretable circadian features add ranking information?",
    },
    {
        "contrast_id": "C4",
        "tier": "secondary",
        "model_id": "M4",
        "reference_id": "M3",
        "question": "Does a linear learned profile add beyond circadian features?",
    },
    {
        "contrast_id": "C5",
        "tier": "secondary",
        "model_id": "M5",
        "reference_id": "M3",
        "question": "Does a nonlinear learned profile add beyond circadian features?",
    },
    {
        "contrast_id": "C6",
        "tier": "exploratory",
        "model_id": "M5",
        "reference_id": "M4",
        "question": "Does nonlinear AE encoding outperform linear PCA encoding?",
    },
    {
        "contrast_id": "C7",
        "tier": "primary",
        "model_id": "M6",
        "reference_id": "M3",
        "question": "Does AE information add beyond the interpretable circadian model?",
    },
)


def interpretation(point: float, low: float, high: float) -> str:
    if low > 0:
        return "clear positive paired gain"
    if high < 0:
        return "clear negative paired change"
    if point > 0:
        return "positive point estimate; uncertainty includes no gain"
    if point < 0:
        return "negative point estimate; uncertainty includes no change"
    return "no observed change"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("conservative_one_se", "maximum_inner_mean"),
        default="conservative_one_se",
    )
    args = parser.parse_args()
    config = load_config(args.smoke)
    if args.smoke and args.policy != "conservative_one_se":
        raise ValueError("Smoke predictions exist only for the conservative one-SE policy")
    if args.smoke:
        prediction_path = PREDICTIONS / "oof_predictions_smoke.parquet"
        output_name = "prespecified_comparisons_smoke.csv"
    elif args.policy == "maximum_inner_mean":
        prediction_path = PREDICTIONS / "oof_predictions_max_mean.parquet"
        output_name = "prespecified_comparisons_max_mean.csv"
    else:
        prediction_path = PREDICTIONS / "oof_predictions.parquet"
        output_name = "prespecified_comparisons.csv"
    predictions = pd.read_parquet(prediction_path)
    participants = (
        predictions[["SEQN", "cycle", "outer_fold", "time_years", "cvd_death"]]
        .drop_duplicates("SEQN")
        .sort_values("SEQN")
        .reset_index(drop=True)
    )
    wide = (
        predictions.pivot(index="SEQN", columns="model_id", values="linear_predictor")
        .loc[participants.SEQN]
    )
    point = {
        model_id: c_index(participants, wide[model_id].to_numpy(float))
        for model_id in wide.columns
    }
    rng = np.random.default_rng(int(config["seed"]) + 804)
    bootstrap_indices = [
        rng.integers(0, len(participants), len(participants))
        for _ in range(int(config["bootstrap_replicates"]))
    ]
    rows = []
    for definition in CONTRASTS:
        model_id = definition["model_id"]
        reference_id = definition["reference_id"]
        model_values = wide[model_id].to_numpy(float)
        reference_values = wide[reference_id].to_numpy(float)
        deltas = []
        for indices in bootstrap_indices:
            sample = participants.iloc[indices].reset_index(drop=True)
            deltas.append(
                c_index(sample, model_values[indices])
                - c_index(sample, reference_values[indices])
            )
        low, high = np.percentile(deltas, [2.5, 97.5])
        delta = point[model_id] - point[reference_id]
        rows.append(
            {
                **definition,
                "model_label": MODEL_BY_ID[model_id]["label"],
                "reference_label": MODEL_BY_ID[reference_id]["label"],
                "model_c_index": point[model_id],
                "reference_c_index": point[reference_id],
                "delta_c_index": delta,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_probability_positive": float(np.mean(np.asarray(deltas) > 0)),
                "interpretation": interpretation(delta, low, high),
                "multiplicity_note": (
                    "single primary contrast"
                    if definition["tier"] == "primary"
                    else "descriptive; no multiplicity-adjusted claim"
                ),
            }
        )
    output = pd.DataFrame(rows)
    output.insert(1, "tuning_policy", args.policy)
    output.to_csv(TABLES / output_name, index=False)
    print(
        output[
            [
                "contrast_id",
                "tier",
                "model_id",
                "reference_id",
                "delta_c_index",
                "ci_low",
                "ci_high",
                "interpretation",
            ]
        ].round(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
