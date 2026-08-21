"""Run the canonical nested average-day models and save one OOF prediction file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models import (
    CACHE,
    DATA,
    MODEL_BY_ID,
    MODEL_IDS,
    PREDICTIONS,
    TABLES,
    absolute_risk_from_training,
    fit_selected_model,
    load_config,
    software_versions,
    stratified_folds,
)


def load_inputs(smoke: bool, config: dict):
    cohort = pd.read_parquet(DATA / "cohort.parquet")
    with np.load(DATA / "average_day_profiles.npz") as stored:
        seqn = stored["seqn"].astype(int)
        profiles = stored["profile"].astype(np.float32)
    if not np.array_equal(seqn, cohort.SEQN.to_numpy(int)):
        raise ValueError("Cohort and profile participant order disagree")
    if smoke:
        rng = np.random.default_rng(99)
        events = np.flatnonzero(cohort.cvd_death.to_numpy(int) == 1)
        nonevents = np.flatnonzero(cohort.cvd_death.to_numpy(int) == 0)
        selected = np.sort(
            np.concatenate(
                [events, rng.choice(nonevents, int(config["smoke"]["non_events"]), replace=False)]
            )
        )
        cohort = cohort.iloc[selected].reset_index(drop=True)
        profiles = profiles[selected]
        folds = stratified_folds(cohort, int(config["outer_folds"]), int(config["seed"]))
    else:
        assignments = pd.read_csv(TABLES / "fold_assignments.csv").set_index("SEQN")
        fold_values = assignments.loc[cohort.SEQN.astype(int), "outer_fold"].to_numpy(int)
        folds = [np.flatnonzero(fold_values == number) for number in range(1, int(config["outer_folds"]) + 1)]
    return cohort, profiles, folds


def fold_complete(suffix: str, fold_number: int) -> bool:
    prediction = CACHE / f"oof_{suffix}_fold_{fold_number}.parquet"
    features = [CACHE / f"features_{suffix}_fold_{fold_number}_{model_id}.npz" for model_id in MODEL_IDS]
    return prediction.exists() and all(path.exists() for path in features)


def table_output(stem: str, smoke: bool, extension: str = "csv") -> Path:
    suffix = "_smoke" if smoke else ""
    return TABLES / f"{stem}{suffix}.{extension}"


def run_fold(
    data: pd.DataFrame,
    profiles: np.ndarray,
    test_indices: np.ndarray,
    fold_number: int,
    config: dict,
    suffix: str,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_path = CACHE / f"oof_{suffix}_fold_{fold_number}.parquet"
    tuning_path = CACHE / f"tuning_{suffix}_fold_{fold_number}.csv"
    history_path = CACHE / f"ae_history_{suffix}_fold_{fold_number}.csv"
    if not force and fold_complete(suffix, fold_number):
        print(f"Reusing completed outer fold {fold_number}")
        return (
            pd.read_parquet(prediction_path),
            pd.read_csv(tuning_path) if tuning_path.exists() else pd.DataFrame(),
            pd.read_csv(history_path) if history_path.exists() else pd.DataFrame(),
        )
    all_indices = np.arange(len(data))
    train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
    train_data = data.iloc[train_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    train_profiles, test_profiles = profiles[train_indices], profiles[test_indices]
    rows, tuning_rows, history_rows = [], [], []
    print(
        f"Outer fold {fold_number}: train={len(train_indices):,}/{int(train_data.cvd_death.sum())} events; "
        f"test={len(test_indices):,}/{int(test_data.cvd_death.sum())} events",
        flush=True,
    )
    for model_position, model_id in enumerate(MODEL_IDS):
        print(f"  {model_id}: {MODEL_BY_ID[model_id]['label']}", flush=True)
        fitted = fit_selected_model(
            train_data,
            test_data,
            train_profiles,
            test_profiles,
            model_id,
            config,
            int(config["seed"]) + fold_number * 10000 + model_position * 1000,
        )
        selected = fitted["selected"]
        absolute_risk = absolute_risk_from_training(
            train_data,
            fitted["train_lp"],
            fitted["test_lp"],
            float(config["horizon_years"]),
        )
        for local_position, global_index in enumerate(test_indices):
            row = data.iloc[global_index]
            rows.append(
                {
                    "SEQN": int(row.SEQN),
                    "cycle": str(row.cycle),
                    "outer_fold": fold_number,
                    "time_years": float(row.time_years),
                    "cvd_death": int(row.cvd_death),
                    "endpoint": str(row.get("endpoint", config.get("endpoint", {}).get("name", "cvd_death"))),
                    "model_id": model_id,
                    "model_label": MODEL_BY_ID[model_id]["label"],
                    "linear_predictor": float(fitted["test_lp"][local_position]),
                    "risk_10y": float(absolute_risk[local_position]),
                    "selected_dimension": int(selected.dimension),
                    "selected_alpha": float(selected.alpha),
                    "inner_c_index": float(selected.mean_c_index),
                }
            )
        tuning_rows.extend(
            {"outer_fold": fold_number, "model_id": model_id, **record}
            for record in fitted["tuning"]
        )
        history_rows.extend(
            {"outer_fold": fold_number, "model_id": model_id, **record}
            for record in fitted["history"]
        )
        np.savez_compressed(
            CACHE / f"features_{suffix}_fold_{fold_number}_{model_id}.npz",
            train_indices=train_indices,
            test_indices=test_indices,
            train_features=fitted["train_features"],
            test_features=fitted["test_features"],
            selected_dimension=np.asarray([selected.dimension]),
            selected_alpha=np.asarray([selected.alpha]),
            feature_mean=fitted["head"].get("feature_mean", np.asarray([])),
            feature_scale=fitted["head"].get("feature_scale", np.asarray([])),
            beta=fitted["head"].get("beta", np.asarray([])),
        )
    predictions = pd.DataFrame(rows)
    tuning = pd.DataFrame(tuning_rows)
    history = pd.DataFrame(history_rows)
    predictions.to_parquet(prediction_path, index=False)
    tuning.to_csv(tuning_path, index=False)
    history.to_csv(history_path, index=False)
    return predictions, tuning, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.smoke)
    suffix = "smoke" if args.smoke else "full"
    data, profiles, folds = load_inputs(args.smoke, config)
    if args.fold is not None and not 1 <= args.fold <= len(folds):
        raise ValueError(f"--fold must be between 1 and {len(folds)}")
    selected_folds = [args.fold] if args.fold is not None else list(range(1, len(folds) + 1))
    for fold_number in selected_folds:
        run_fold(
            data,
            profiles,
            folds[fold_number - 1],
            fold_number,
            config,
            suffix,
            args.force,
        )
    completed = [number for number in range(1, len(folds) + 1) if fold_complete(suffix, number)]
    if len(completed) != len(folds):
        print(f"Completed folds {completed}; run remaining folds before consolidating predictions")
        return
    predictions = pd.concat(
        [pd.read_parquet(CACHE / f"oof_{suffix}_fold_{number}.parquet") for number in completed],
        ignore_index=True,
    ).sort_values(["SEQN", "model_id"]).reset_index(drop=True)
    predictions["model_label"] = predictions.model_id.map(
        lambda value: MODEL_BY_ID[value]["label"]
    )
    tuning = pd.concat(
        [pd.read_csv(CACHE / f"tuning_{suffix}_fold_{number}.csv") for number in completed],
        ignore_index=True,
    )
    histories = [
        pd.read_csv(CACHE / f"ae_history_{suffix}_fold_{number}.csv")
        for number in completed
        if (CACHE / f"ae_history_{suffix}_fold_{number}.csv").stat().st_size > 1
    ]
    output = PREDICTIONS / ("oof_predictions_smoke.parquet" if args.smoke else "oof_predictions.parquet")
    predictions.to_parquet(output, index=False)
    tuning.to_csv(table_output("nested_tuning", args.smoke), index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(
            table_output("ae_training_history", args.smoke), index=False
        )
    selections = (
        predictions[["outer_fold", "model_id", "selected_dimension", "selected_alpha", "inner_c_index"]]
        .drop_duplicates()
        .sort_values(["model_id", "outer_fold"])
    )
    selections.to_csv(table_output("nested_selections", args.smoke), index=False)
    metadata = {
        "run": suffix,
        "participants": int(predictions.SEQN.nunique()),
        "events": int(data.cvd_death.sum()),
        "endpoint": config.get("endpoint", {}),
        "outer_folds": len(folds),
        "inner_folds": int(config["inner_folds"]),
        "dimensions": config["dimensions"],
        "ridge_alphas": config["ridge_alphas"],
        "seed": config["seed"],
        "software": software_versions(),
    }
    with table_output("run_metadata", args.smoke, "json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved canonical OOF predictions: {output}")


if __name__ == "__main__":
    main()
