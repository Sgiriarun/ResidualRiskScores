"""Strict outer-fold comparison of fixed AE dimensions, with no test-set tuning."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from models import (
    CACHE,
    DATA,
    PREDICTIONS,
    TABLES,
    c_index,
    fit_head,
    fit_selected_model,
    load_config,
)


def bootstrap_summary(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    references: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (model_id, dimension, strategy), group in predictions.groupby(
        ["model_id", "dimension", "strategy"], sort=True
    ):
        risk = group.set_index("SEQN").loc[data.SEQN, "linear_predictor"].to_numpy(float)
        point = c_index(data, risk)
        point_base = c_index(data, references["M0"])
        point_circadian = c_index(data, references["M3"])
        values, delta_base, delta_circadian = [], [], []
        for _ in range(replicates):
            selected = rng.integers(0, len(data), len(data))
            sample = data.iloc[selected].reset_index(drop=True)
            score = c_index(sample, risk[selected])
            base_score = c_index(sample, references["M0"][selected])
            circadian_score = c_index(sample, references["M3"][selected])
            values.append(score)
            delta_base.append(score - base_score)
            delta_circadian.append(score - circadian_score)
        low, high = np.percentile(values, [2.5, 97.5])
        base_low, base_high = np.percentile(delta_base, [2.5, 97.5])
        circ_low, circ_high = np.percentile(delta_circadian, [2.5, 97.5])
        rows.append(
            {
                "model_id": model_id,
                "dimension": int(dimension),
                "strategy": strategy,
                "c_index": point,
                "ci_low": low,
                "ci_high": high,
                "delta_vs_M0": point - point_base,
                "delta_vs_M0_low": base_low,
                "delta_vs_M0_high": base_high,
                "delta_vs_M3": point - point_circadian,
                "delta_vs_M3_low": circ_low,
                "delta_vs_M3_high": circ_high,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    data = pd.read_parquet(DATA / "cohort.parquet")
    with np.load(DATA / "average_day_profiles.npz") as stored:
        profiles = stored["profile"].astype(np.float32)
    assignments = pd.read_csv(TABLES / "fold_assignments.csv").set_index("SEQN")
    fold_values = assignments.loc[data.SEQN, "outer_fold"].to_numpy(int)
    canonical = pd.read_parquet(PREDICTIONS / "oof_predictions.parquet")
    canonical_wide = canonical.pivot(index="SEQN", columns="model_id", values="linear_predictor").loc[data.SEQN]
    references = {model_id: canonical_wide[model_id].to_numpy(float) for model_id in ("M0", "M3")}
    prediction_rows, selection_rows = [], []
    all_indices = np.arange(len(data))
    for model_position, model_id in enumerate(("M5", "M6")):
        for dimension in config["dimensions"]:
            dimension_config = {**config, "dimensions": [int(dimension)]}
            for fold_number in range(1, int(config["outer_folds"]) + 1):
                cache_path = CACHE / f"dimension_{model_id}_{dimension}_fold_{fold_number}.npz"
                test_indices = np.flatnonzero(fold_values == fold_number)
                train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
                if cache_path.exists() and not args.force:
                    with np.load(cache_path) as stored:
                        nested_lp = stored["nested_lp"]
                        fixed_lp = stored["fixed_alpha_lp"]
                        selected_alpha = float(stored["selected_alpha"][0])
                        inner_c_index = float(stored["inner_c_index"][0])
                else:
                    print(f"{model_id}, dimension={dimension}, fold={fold_number}", flush=True)
                    train_data = data.iloc[train_indices].reset_index(drop=True)
                    test_data = data.iloc[test_indices].reset_index(drop=True)
                    fitted = fit_selected_model(
                        train_data,
                        test_data,
                        profiles[train_indices],
                        profiles[test_indices],
                        model_id,
                        dimension_config,
                        int(config["seed"]) + 60000 + model_position * 10000 + dimension * 100 + fold_number,
                    )
                    nested_lp = fitted["test_lp"]
                    _, fixed_lp, _ = fit_head(
                        train_data,
                        test_data,
                        fitted["train_features"],
                        fitted["test_features"],
                        0.1,
                    )
                    selected_alpha = float(fitted["selected"].alpha)
                    inner_c_index = float(fitted["selected"].mean_c_index)
                    np.savez_compressed(
                        cache_path,
                        test_indices=test_indices,
                        nested_lp=nested_lp,
                        fixed_alpha_lp=fixed_lp,
                        selected_alpha=np.asarray([selected_alpha]),
                        inner_c_index=np.asarray([inner_c_index]),
                    )
                selection_rows.append(
                    {
                        "model_id": model_id,
                        "dimension": dimension,
                        "outer_fold": fold_number,
                        "selected_alpha": selected_alpha,
                        "inner_c_index": inner_c_index,
                    }
                )
                for strategy, values in (("nested_alpha", nested_lp), ("fixed_alpha_0.1", fixed_lp)):
                    for local, index in enumerate(test_indices):
                        prediction_rows.append(
                            {
                                "SEQN": int(data.iloc[index].SEQN),
                                "model_id": model_id,
                                "dimension": dimension,
                                "strategy": strategy,
                                "outer_fold": fold_number,
                                "linear_predictor": float(values[local]),
                            }
                        )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_parquet(PREDICTIONS / "dimension_sensitivity_predictions.parquet", index=False)
    pd.DataFrame(selection_rows).to_csv(TABLES / "dimension_sensitivity_selections.csv", index=False)
    summary = bootstrap_summary(
        data,
        predictions,
        references,
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + 603,
    )
    summary.to_csv(TABLES / "dimension_sensitivity.csv", index=False)
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()

