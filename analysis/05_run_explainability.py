"""Cross-validated AE readouts and out-of-fold correction phenotypes."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from models import CACHE, DATA, PREDICTIONS, TABLES, ae_transform, load_config

KNOWN_COLUMNS = ["volume_mean_cpm", "mvpa_min_day", "M10", "L5", "RA"]
KNOWN_LABELS = ["Volume", "MVPA", "M10", "L5", "RA"]


def output_name(stem: str, smoke: bool) -> str:
    return f"{stem}_smoke.csv" if smoke else f"{stem}.csv"


def load_inputs(smoke: bool):
    data = pd.read_parquet(DATA / "cohort.parquet")
    with np.load(DATA / "average_day_profiles.npz") as stored:
        profiles = stored["profile"].astype(np.float32)
    predictions = pd.read_parquet(
        PREDICTIONS / ("oof_predictions_smoke.parquet" if smoke else "oof_predictions.parquet")
    )
    if smoke:
        keep = np.sort(predictions.SEQN.unique())
        positions = data.set_index("SEQN").index.get_indexer(keep)
        data = data.iloc[positions].reset_index(drop=True)
        profiles = profiles[positions]
    return data, profiles, predictions


def ridge_readout(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    x_mean, x_scale = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    y_mean, y_scale = train_y.mean(axis=0), train_y.std(axis=0) + 1e-6
    train_z = (train_x - x_mean) / x_scale
    test_z = (test_x - x_mean) / x_scale
    target_z = (train_y - y_mean) / y_scale
    design = np.column_stack([np.ones(len(train_z)), train_z])
    penalty = np.eye(design.shape[1]) * 1e-3
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ target_z)
    prediction_z = np.column_stack([np.ones(len(test_z)), test_z]) @ beta
    return prediction_z * y_scale + y_mean


def cross_validated_readout(data: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    suffix = "smoke" if smoke else "full"
    predicted = np.full((len(data), len(KNOWN_COLUMNS)), np.nan)
    fold_count = int(data.outer_fold.nunique()) if "outer_fold" in data else None
    cached = sorted(CACHE.glob(f"features_{suffix}_fold_*_M5.npz"))
    if not cached:
        raise FileNotFoundError("Missing M5 fold features; run 02_run_nested_models.py")
    targets = data[KNOWN_COLUMNS].to_numpy(float)
    for path in cached:
        with np.load(path) as stored:
            train_indices = stored["train_indices"].astype(int)
            test_indices = stored["test_indices"].astype(int)
            train_features = stored["train_features"]
            test_features = stored["test_features"]
        predicted[test_indices] = ridge_readout(
            train_features, test_features, targets[train_indices]
        )
    rows = []
    for column, label in enumerate(KNOWN_LABELS):
        observed = targets[:, column]
        residual = np.sum((observed - predicted[:, column]) ** 2)
        total = np.sum((observed - observed.mean()) ** 2)
        rows.append(
            {
                "feature": label,
                "cross_validated_R2": float(1 - residual / total),
                "participants": int(np.isfinite(predicted[:, column]).sum()),
            }
        )
    return pd.DataFrame(rows)


def correction_phenotypes(
    data: pd.DataFrame,
    profiles: np.ndarray,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = predictions.pivot(index="SEQN", columns="model_id", values="linear_predictor").loc[data.SEQN]
    correction = wide.M5.to_numpy(float) - wide.M0.to_numpy(float)
    group = pd.qcut(pd.Series(correction).rank(method="first"), 10, labels=False).to_numpy()
    low, high = group == 0, group == 9
    known = data[KNOWN_COLUMNS].to_numpy(float)
    scale = known.std(axis=0, ddof=1) + 1e-9
    difference = (known[high].mean(axis=0) - known[low].mean(axis=0)) / scale
    feature_table = pd.DataFrame(
        {
            "feature": KNOWN_LABELS,
            "high_minus_low_correction_group_SD": difference,
            "low_group_mean": known[low].mean(axis=0),
            "high_group_mean": known[high].mean(axis=0),
        }
    )
    curve_rows = []
    for minute in range(profiles.shape[1]):
        row = {"minute": minute, "clock_hour": minute / 60}
        for label, mask in (("low", low), ("high", high)):
            values = profiles[mask, minute]
            standard_error = values.std(ddof=1) / np.sqrt(len(values))
            row[f"{label}_mean"] = float(values.mean())
            row[f"{label}_ci_low"] = float(values.mean() - 1.96 * standard_error)
            row[f"{label}_ci_high"] = float(values.mean() + 1.96 * standard_error)
        curve_rows.append(row)
    group_table = pd.DataFrame(
        [
            {
                "group": "lowest_correction_decile",
                "participants": int(low.sum()),
                "mean_AE_correction": float(correction[low].mean()),
            },
            {
                "group": "highest_correction_decile",
                "participants": int(high.sum()),
                "mean_AE_correction": float(correction[high].mean()),
            },
        ]
    )
    return feature_table, pd.DataFrame(curve_rows), group_table


def descriptive_latent_correlations(
    data: pd.DataFrame,
    profiles: np.ndarray,
    predictions: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    selected = (
        predictions[predictions.model_id == "M5"]
        [["outer_fold", "selected_dimension"]]
        .drop_duplicates()
    )
    dimension = int(selected.selected_dimension.mode().iloc[0])
    embedding, _, _ = ae_transform(
        profiles,
        profiles,
        dimension,
        config["ae"],
        int(config["seed"]) + 50000,
    )
    known = data[KNOWN_COLUMNS].to_numpy(float)
    rows = []
    for latent in range(dimension):
        for column, label in enumerate(KNOWN_LABELS):
            rows.append(
                {
                    "latent_dimension": f"emb{latent}",
                    "known_feature": label,
                    "pearson_r": float(np.corrcoef(embedding[:, latent], known[:, column])[0, 1]),
                    "analysis_type": "descriptive_full_cohort_not_performance_validation",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.smoke)
    data, profiles, predictions = load_inputs(args.smoke)
    readout = cross_validated_readout(data, args.smoke)
    features, curves, groups = correction_phenotypes(data, profiles, predictions)
    correlations = descriptive_latent_correlations(data, profiles, predictions, config)
    readout.to_csv(TABLES / output_name("ae_readout_r2", args.smoke), index=False)
    features.to_csv(TABLES / output_name("ae_correction_known_features", args.smoke), index=False)
    curves.to_csv(TABLES / output_name("ae_correction_profiles", args.smoke), index=False)
    groups.to_csv(TABLES / output_name("ae_correction_groups", args.smoke), index=False)
    correlations.to_csv(TABLES / output_name("descriptive_latent_correlations", args.smoke), index=False)
    print(readout.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

