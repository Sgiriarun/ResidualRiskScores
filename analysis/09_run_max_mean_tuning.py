"""Leakage-free sensitivity using maximum mean inner C-index for tuning."""

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
from models import (
    CACHE,
    DATA,
    MODEL_BY_ID,
    MODEL_IDS,
    PREDICTIONS,
    TABLES,
    absolute_risk_from_training,
    fit_head,
    load_config,
    representation_features,
)


LEARNED_MODELS = ("M4", "M5", "M6")


def choose_max_mean(tuning: pd.DataFrame, fold: int, model_id: str) -> pd.Series:
    candidates = tuning[(tuning.outer_fold == fold) & (tuning.model_id == model_id)].copy()
    if candidates.empty:
        raise ValueError(f"No saved inner tuning for fold={fold}, model={model_id}")
    return candidates.sort_values(
        ["mean_c_index", "dimension", "alpha"],
        ascending=[False, True, False],
    ).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    data = pd.read_parquet(DATA / "cohort.parquet")
    with np.load(DATA / "average_day_profiles.npz") as stored:
        profiles = stored["profile"].astype(np.float32)
    canonical = pd.read_parquet(PREDICTIONS / "oof_predictions.parquet")
    tuning = pd.read_csv(TABLES / "nested_tuning.csv")
    assignments = pd.read_csv(TABLES / "fold_assignments.csv").set_index("SEQN")
    fold_values = assignments.loc[data.SEQN, "outer_fold"].to_numpy(int)
    rows = []
    selections = []
    all_indices = np.arange(len(data))

    fixed = canonical[~canonical.model_id.isin(LEARNED_MODELS)].copy()
    rows.extend(fixed.to_dict("records"))
    for fold in range(1, int(config["outer_folds"]) + 1):
        test_indices = np.flatnonzero(fold_values == fold)
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        train_data = data.iloc[train_indices].reset_index(drop=True)
        test_data = data.iloc[test_indices].reset_index(drop=True)
        for model_id in LEARNED_MODELS:
            chosen = choose_max_mean(tuning, fold, model_id)
            dimension = int(chosen.dimension)
            alpha = float(chosen.alpha)
            model_position = MODEL_IDS.index(model_id)
            seed = int(config["seed"]) + fold * 10000 + model_position * 1000 + 5000
            cache_path = CACHE / f"max_mean_fold_{fold}_{model_id}.npz"
            if cache_path.exists() and not args.force:
                with np.load(cache_path) as stored:
                    train_lp = stored["train_lp"]
                    test_lp = stored["test_lp"]
            else:
                print(
                    f"fold={fold} {model_id}: dimension={dimension}, alpha={alpha:g}",
                    flush=True,
                )
                train_features, test_features, _ = representation_features(
                    model_id,
                    train_data,
                    test_data,
                    profiles[train_indices],
                    profiles[test_indices],
                    dimension,
                    config,
                    seed,
                )
                train_lp, test_lp, _ = fit_head(
                    train_data,
                    test_data,
                    train_features,
                    test_features,
                    alpha,
                )
                np.savez_compressed(cache_path, train_lp=train_lp, test_lp=test_lp)
            risk = absolute_risk_from_training(
                train_data,
                train_lp,
                test_lp,
                float(config["horizon_years"]),
            )
            selections.append(
                {
                    "outer_fold": fold,
                    "model_id": model_id,
                    "selected_dimension": dimension,
                    "selected_alpha": alpha,
                    "inner_c_index": float(chosen.mean_c_index),
                    "selection_rule": "maximum_mean_inner_c_index",
                }
            )
            for local, global_index in enumerate(test_indices):
                participant = data.iloc[global_index]
                rows.append(
                    {
                        "SEQN": int(participant.SEQN),
                        "cycle": str(participant.cycle),
                        "outer_fold": fold,
                        "time_years": float(participant.time_years),
                        "cvd_death": int(participant.cvd_death),
                        "endpoint": str(participant.get("endpoint", config.get("endpoint", {}).get("name", "cvd_death"))),
                        "model_id": model_id,
                        "model_label": MODEL_BY_ID[model_id]["label"],
                        "linear_predictor": float(test_lp[local]),
                        "risk_10y": float(risk[local]),
                        "selected_dimension": dimension,
                        "selected_alpha": alpha,
                        "inner_c_index": float(chosen.mean_c_index),
                    }
                )

    predictions = pd.DataFrame(rows).sort_values(["SEQN", "model_id"]).reset_index(drop=True)
    if len(predictions) != len(data) * len(MODEL_IDS):
        raise AssertionError("Maximum-mean prediction table is incomplete")
    predictions.to_parquet(PREDICTIONS / "oof_predictions_max_mean.parquet", index=False)
    pd.DataFrame(selections).to_csv(TABLES / "max_mean_selections.csv", index=False)

    participants = (
        predictions[["SEQN", "cycle", "outer_fold", "time_years", "cvd_death", "endpoint"]]
        .drop_duplicates("SEQN")
        .sort_values("SEQN")
        .reset_index(drop=True)
    )
    lp_wide = predictions.pivot(index="SEQN", columns="model_id", values="linear_predictor").loc[participants.SEQN]
    risk_wide = predictions.pivot(index="SEQN", columns="model_id", values="risk_10y").loc[participants.SEQN]
    linear_predictors = {model_id: lp_wide[model_id].to_numpy(float) for model_id in MODEL_IDS}
    risks = {model_id: risk_wide[model_id].to_numpy(float) for model_id in MODEL_IDS}
    discrimination = bootstrap_discrimination(
        participants,
        linear_predictors,
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + 905,
    )
    discrimination["model_label"] = discrimination.model_id.map(
        lambda value: MODEL_BY_ID[value]["label"]
    )
    clinical = bootstrap_clinical_metrics(
        participants,
        linear_predictors,
        risks,
        float(config["horizon_years"]),
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + 906,
    )
    clinical["model_label"] = clinical.model_id.map(lambda value: MODEL_BY_ID[value]["label"])
    discrimination.to_csv(TABLES / "max_mean_primary_results.csv", index=False)
    clinical.to_csv(TABLES / "max_mean_clinical_metrics.csv", index=False)
    decision_curve(participants, risks, float(config["horizon_years"])).to_csv(
        TABLES / "max_mean_decision_curve.csv", index=False
    )
    calibration_table(participants, risks, float(config["horizon_years"])).to_csv(
        TABLES / "max_mean_calibration_by_decile.csv", index=False
    )
    print(
        discrimination[
            ["model_id", "c_index", "delta_vs_M0", "delta_vs_M3_low", "delta_vs_M3", "delta_vs_M3_high"]
        ].round(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
