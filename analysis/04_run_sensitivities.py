"""Run endpoint, mapping, cycle-transport, and competing-mortality checks."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import aalen_johansen_cif, bootstrap_discrimination
from models import (
    CACHE,
    DATA,
    MODEL_BY_ID,
    MODEL_IDS,
    PREDICTIONS,
    TABLES,
    c_index,
    fit_head,
    fit_selected_model,
    load_config,
    stratified_folds,
)

MORTALITY_FILES = {
    "C": "NHANES_2003_2004_MORT_2019_PUBLIC.dat",
    "D": "NHANES_2005_2006_MORT_2019_PUBLIC.dat",
}
MORT_BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"


def output_name(stem: str, smoke: bool) -> str:
    return f"{stem}_smoke.csv" if smoke else f"{stem}.csv"


def load_analysis(smoke: bool, config: dict):
    data = pd.read_parquet(DATA / "cohort.parquet")
    with np.load(DATA / "average_day_profiles.npz") as stored:
        profiles = stored["profile"].astype(np.float32)
    prediction_path = PREDICTIONS / ("oof_predictions_smoke.parquet" if smoke else "oof_predictions.parquet")
    predictions = pd.read_parquet(prediction_path)
    if smoke:
        keep = np.sort(predictions.SEQN.unique())
        position = data.set_index("SEQN").index.get_indexer(keep)
        if (position < 0).any():
            raise ValueError("Smoke participants are absent from the canonical cohort")
        data = data.iloc[position].reset_index(drop=True)
        profiles = profiles[position]
    return data, profiles, predictions


def read_mortality() -> pd.DataFrame:
    rows = []
    for cycle, filename in MORTALITY_FILES.items():
        candidates = [Path("/private/tmp") / filename, Path("/tmp") / filename]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            path = Path("/private/tmp") / filename
            urllib.request.urlretrieve(MORT_BASE + filename, path)
        mortality = pd.read_fwf(
            path,
            colspecs=[(0, 6), (15, 16), (16, 19)],
            names=["SEQN", "MORTSTAT", "UCOD_LEADING"],
            dtype={"UCOD_LEADING": str},
            na_values=["."],
        )
        mortality["SEQN"] = mortality.SEQN.astype(int)
        mortality["cycle"] = cycle
        rows.append(mortality)
    return pd.concat(rows, ignore_index=True)


def prediction_wide(predictions: pd.DataFrame, participant_order: np.ndarray) -> dict[str, np.ndarray]:
    wide = predictions.pivot(index="SEQN", columns="model_id", values="linear_predictor").loc[participant_order]
    return {model_id: wide[model_id].to_numpy(float) for model_id in MODEL_IDS}


def endpoint_refit(data: pd.DataFrame, mortality: pd.DataFrame, smoke: bool, config: dict) -> pd.DataFrame:
    sensitivity = config.get("sensitivity_endpoints", {}).get(
        "heart_only",
        {"codes": ["001"], "name": "heart_disease_death"},
    )
    codes = [str(code) for code in sensitivity["codes"]]
    endpoint = mortality.copy()
    endpoint["cvd_death"] = (
        (pd.to_numeric(endpoint.MORTSTAT, errors="coerce") == 1)
        & endpoint.UCOD_LEADING.astype(str).isin(codes)
    ).astype(int)
    aligned = data.drop(columns="cvd_death").merge(
        endpoint[["SEQN", "cycle", "cvd_death"]],
        on=["SEQN", "cycle"],
        how="left",
        validate="one_to_one",
    )
    if aligned.cvd_death.isna().any():
        raise ValueError("Sensitivity endpoint labels are missing")
    suffix = "smoke" if smoke else "full"
    predictors = {model_id: np.full(len(aligned), np.nan) for model_id in MODEL_IDS}
    folds = sorted(pd.read_parquet(CACHE / f"oof_{suffix}_fold_1.parquet").outer_fold.unique())
    fold_count = int(config["outer_folds"])
    for fold_number in range(1, fold_count + 1):
        for model_id in MODEL_IDS:
            with np.load(CACHE / f"features_{suffix}_fold_{fold_number}_{model_id}.npz") as stored:
                train_indices = stored["train_indices"].astype(int)
                test_indices = stored["test_indices"].astype(int)
                alpha = float(stored["selected_alpha"][0])
                train_features = stored["train_features"]
                test_features = stored["test_features"]
            train_data = aligned.iloc[train_indices].reset_index(drop=True)
            test_data = aligned.iloc[test_indices].reset_index(drop=True)
            if model_id == "M0":
                predictors[model_id][test_indices] = test_data.lp_predict.to_numpy(float)
            else:
                _, test_lp, _ = fit_head(
                    train_data, test_data, train_features, test_features, alpha
                )
                predictors[model_id][test_indices] = test_lp
    result = bootstrap_discrimination(
        aligned,
        predictors,
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + 801,
    )
    result["endpoint"] = str(sensitivity["name"])
    return result


def direct_mapping_sensitivity(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    mapped = data.has_predict_cat.eq(1).to_numpy()
    subset = data.loc[mapped].reset_index(drop=True)
    predictors = prediction_wide(predictions, data.SEQN.to_numpy(int))
    predictors = {model_id: values[mapped] for model_id, values in predictors.items()}
    result = bootstrap_discrimination(
        subset,
        predictors,
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + 802,
    )
    result["participants"] = len(subset)
    result["events"] = int(subset.cvd_death.sum())
    result["analysis"] = "direct_PREDICT_ethnicity_mapping_subgroup"
    return result


def transport_analysis(
    data: pd.DataFrame,
    profiles: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    rows = []
    for train_cycle, test_cycle in (("C", "D"), ("D", "C")):
        train_indices = np.flatnonzero(data.cycle.to_numpy(str) == train_cycle)
        test_indices = np.flatnonzero(data.cycle.to_numpy(str) == test_cycle)
        train_data = data.iloc[train_indices].reset_index(drop=True)
        test_data = data.iloc[test_indices].reset_index(drop=True)
        predictors = {}
        selections = {}
        for position, model_id in enumerate(MODEL_IDS):
            print(f"Transport {train_cycle}->{test_cycle}: {model_id}", flush=True)
            fitted = fit_selected_model(
                train_data,
                test_data,
                profiles[train_indices],
                profiles[test_indices],
                model_id,
                config,
                int(config["seed"]) + 20000 + ord(train_cycle) * 100 + position * 1000,
            )
            predictors[model_id] = fitted["test_lp"]
            selections[model_id] = fitted["selected"]
        bootstrap = bootstrap_discrimination(
            test_data,
            predictors,
            int(config["bootstrap_replicates"]),
            int(config["seed"]) + 900 + ord(train_cycle),
        )
        for record in bootstrap.to_dict("records"):
            selected = selections[record["model_id"]]
            rows.append(
                {
                    "train_cycle": train_cycle,
                    "test_cycle": test_cycle,
                    "test_participants": len(test_data),
                    "test_events": int(test_data.cvd_death.sum()),
                    "selected_dimension": selected.dimension,
                    "selected_alpha": selected.alpha,
                    **record,
                }
            )
    return pd.DataFrame(rows)


def competing_mortality_table(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    mortality: pd.DataFrame,
    horizon: float,
    config: dict,
) -> pd.DataFrame:
    status = data[["SEQN", "cycle"]].merge(
        mortality, on=["SEQN", "cycle"], how="left", validate="one_to_one"
    )
    died = pd.to_numeric(status.MORTSTAT, errors="coerce").fillna(0).to_numpy(int) == 1
    primary_codes = set(str(code) for code in config.get("endpoint", {}).get("codes", ["001"]))
    primary_event = pd.Series(status.UCOD_LEADING.astype(str)).isin(primary_codes).to_numpy()
    event_type = np.zeros(len(data), dtype=int)
    event_type[died & primary_event] = 1
    event_type[died & ~primary_event] = 2
    risk_wide = predictions.pivot(index="SEQN", columns="model_id", values="risk_10y").loc[data.SEQN]
    rows = []
    for model_id in MODEL_IDS:
        risk = risk_wide[model_id].to_numpy(float)
        decile = pd.qcut(pd.Series(risk).rank(method="first"), 10, labels=False).to_numpy()
        for group in range(10):
            mask = decile == group
            rows.append(
                {
                    "model_id": model_id,
                    "risk_decile": group + 1,
                    "participants": int(mask.sum()),
                    "mean_predicted_10y_risk": float(risk[mask].mean()),
                    "observed_10y_primary_death_cif": aalen_johansen_cif(
                        data.time_years.to_numpy(float), event_type, horizon, mask
                    ),
                    "non_cvd_deaths": int(((event_type == 2) & mask).sum()),
                }
            )
    return pd.DataFrame(rows)


def event_count_sensitivity(
    data: pd.DataFrame,
    profiles: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    """Optional expensive check that refits every stage within every subsample."""
    event_indices = np.flatnonzero(data.cvd_death.to_numpy(int) == 1)
    nonevent_indices = np.flatnonzero(data.cvd_death.to_numpy(int) == 0)
    event_rate = len(event_indices) / len(data)
    rows = []
    for event_count in config["event_count_grid"]:
        if event_count > len(event_indices):
            continue
        nonevent_count = min(
            len(nonevent_indices), int(round(event_count * (1 - event_rate) / event_rate))
        )
        for repeat in range(int(config["event_count_repeats"])):
            rng = np.random.default_rng(int(config["seed"]) + 30000 + event_count * 10 + repeat)
            selected = np.sort(
                np.concatenate(
                    [
                        rng.choice(event_indices, event_count, replace=False),
                        rng.choice(nonevent_indices, nonevent_count, replace=False),
                    ]
                )
            )
            subset = data.iloc[selected].reset_index(drop=True)
            subset_profiles = profiles[selected]
            folds = stratified_folds(subset, 3, int(config["seed"]) + repeat)
            predictors = {model_id: np.full(len(subset), np.nan) for model_id in MODEL_IDS}
            all_indices = np.arange(len(subset))
            for fold_number, test_indices in enumerate(folds, start=1):
                train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
                train_data = subset.iloc[train_indices].reset_index(drop=True)
                test_data = subset.iloc[test_indices].reset_index(drop=True)
                for position, model_id in enumerate(MODEL_IDS):
                    fitted = fit_selected_model(
                        train_data,
                        test_data,
                        subset_profiles[train_indices],
                        subset_profiles[test_indices],
                        model_id,
                        config,
                        int(config["seed"]) + 40000 + event_count * 100 + repeat * 10 + fold_number + position * 1000,
                    )
                    predictors[model_id][test_indices] = fitted["test_lp"]
            for model_id in MODEL_IDS:
                rows.append(
                    {
                        "target_events": event_count,
                        "actual_events": int(subset.cvd_death.sum()),
                        "participants": len(subset),
                        "repeat": repeat + 1,
                        "model_id": model_id,
                        "oof_c_index": c_index(subset, predictors[model_id]),
                    }
                )
            print(f"Event-count sensitivity: events={event_count}, repeat={repeat + 1}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-transport", action="store_true")
    parser.add_argument("--event-count", action="store_true")
    args = parser.parse_args()
    config = load_config(args.smoke)
    data, profiles, predictions = load_analysis(args.smoke, config)
    mortality = read_mortality()
    endpoint_refit(data, mortality, args.smoke, config).to_csv(
        TABLES / output_name("endpoint_sensitivity", args.smoke), index=False
    )
    direct_mapping_sensitivity(data, predictions, config).to_csv(
        TABLES / output_name("direct_mapping_sensitivity", args.smoke), index=False
    )
    competing_mortality_table(
        data, predictions, mortality, float(config["horizon_years"]), config
    ).to_csv(TABLES / output_name("competing_mortality", args.smoke), index=False)
    if not args.skip_transport:
        transport_analysis(data, profiles, config).to_csv(
            TABLES / output_name("cycle_transport", args.smoke), index=False
        )
    if args.event_count:
        event_count_sensitivity(data, profiles, config).to_csv(
            TABLES / output_name("event_count_sensitivity", args.smoke), index=False
        )
    print("Sensitivity tables generated")


if __name__ == "__main__":
    main()
