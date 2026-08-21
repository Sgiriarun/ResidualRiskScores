"""Shared data contracts, representations, nested selection, and Cox models."""

from __future__ import annotations

import copy
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from scipy.optimize import minimize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / "cache"
PREDICTIONS = HERE / "predictions"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
for directory in (DATA, CACHE, PREDICTIONS, TABLES, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)

MINUTES = 1440
PROFILE_COLUMNS = [f"m{i}" for i in range(MINUTES)]

MODEL_REGISTRY = (
    {"model_id": "M0", "label": "Transported PREDICT LP", "representation": "predict", "purpose": "Clinical baseline"},
    {"model_id": "M1", "label": "PREDICT LP + total volume", "representation": "volume", "purpose": "Overall activity baseline"},
    {"model_id": "M2", "label": "PREDICT LP + MVPA", "representation": "mvpa", "purpose": "Intensity baseline"},
    {"model_id": "M3", "label": "PREDICT LP + M10/L5/RA", "representation": "circadian", "purpose": "Interpretable daily rhythm"},
    {"model_id": "M4", "label": "PREDICT LP + nested PCA", "representation": "pca", "purpose": "Linear learned profile"},
    {"model_id": "M5", "label": "PREDICT LP + nested autoencoder", "representation": "ae", "purpose": "Nonlinear learned profile"},
    {"model_id": "M6", "label": "PREDICT LP + circadian + autoencoder", "representation": "circadian_ae", "purpose": "AI beyond circadian"},
)
MODEL_BY_ID = {row["model_id"]: row for row in MODEL_REGISTRY}
MODEL_IDS = tuple(MODEL_BY_ID)


def load_config(smoke: bool = False) -> dict:
    """Load the JSON-compatible YAML configuration without a PyYAML dependency."""
    with (HERE / "config.yaml").open() as handle:
        config = json.load(handle)
    config["smoke_mode"] = bool(smoke)
    if smoke:
        small = config["smoke"]
        config["outer_folds"] = small["outer_folds"]
        config["inner_folds"] = small["inner_folds"]
        config["dimensions"] = small["dimensions"]
        config["ridge_alphas"] = small["ridge_alphas"]
        config["bootstrap_replicates"] = small["bootstrap_replicates"]
        config["ae"]["epochs"] = small["ae_epochs"]
        config["ae"]["patience"] = 1
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def c_index(data: pd.DataFrame, linear_predictor: np.ndarray) -> float:
    keep = np.isfinite(linear_predictor)
    return float(
        concordance_index(
            data.time_years.to_numpy(float)[keep],
            -np.asarray(linear_predictor, float)[keep],
            data.cvd_death.to_numpy(int)[keep],
        )
    )


def stratified_folds(data: pd.DataFrame, folds: int, seed: int) -> list[np.ndarray]:
    """Round-robin folds stratified jointly by NHANES cycle and event."""
    rng = np.random.default_rng(seed)
    result: list[list[int]] = [[] for _ in range(folds)]
    strata = data.cycle.astype(str) + "_" + data.cvd_death.astype(str)
    for stratum in sorted(strata.unique()):
        indices = np.flatnonzero(strata.to_numpy() == stratum)
        rng.shuffle(indices)
        offset = int(rng.integers(0, folds))
        for position, index in enumerate(indices):
            result[(position + offset) % folds].append(int(index))
    return [np.asarray(sorted(values), dtype=int) for values in result]


def cyclic_interpolate(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fill unobserved clock minutes from neighbouring observed minutes, not zero."""
    values = np.asarray(values, float)
    observed_indices = np.flatnonzero(observed & np.isfinite(values))
    if len(observed_indices) == 0:
        raise ValueError("Average-day profile has no observed wear minutes")
    if len(observed_indices) == MINUTES:
        return values.copy()
    extended_x = np.concatenate([observed_indices - MINUTES, observed_indices, observed_indices + MINUTES])
    extended_y = np.tile(values[observed_indices], 3)
    return np.interp(np.arange(MINUTES), extended_x, extended_y)


def circadian_features(profiles: np.ndarray) -> np.ndarray:
    """Circular M10, L5, and relative amplitude from complete average days."""
    profiles = np.asarray(profiles, float)
    doubled = np.concatenate([profiles, profiles], axis=1)
    cumulative = np.concatenate([np.zeros((len(profiles), 1)), np.cumsum(doubled, axis=1)], axis=1)

    def windows(length: int) -> np.ndarray:
        starts = np.arange(MINUTES)
        return (cumulative[:, starts + length] - cumulative[:, starts]) / length

    m10 = windows(600).max(axis=1)
    l5 = windows(300).min(axis=1)
    ra = (m10 - l5) / (m10 + l5 + 1e-9)
    return np.column_stack([m10, l5, ra])


def standardizer_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return mean, scale


def standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(values, float) - mean) / scale


def pca_transform(train: np.ndarray, test: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = standardizer_fit(train)
    train_z = standardize(train, mean, scale)
    test_z = standardize(test, mean, scale)
    center = train_z.mean(axis=0)
    _, _, vt = np.linalg.svd(train_z - center, full_matrices=False)
    basis = vt[:dimension].T
    return (train_z - center) @ basis, (test_z - center) @ basis


class AverageDayAutoencoder(nn.Module):
    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(MINUTES, hidden), nn.ReLU(), nn.Linear(hidden, dimension))
        self.decoder = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, MINUTES))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(values)
        return self.decoder(embedding), embedding


def _loader(values: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.as_tensor(values, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def ae_transform(
    train: np.ndarray,
    test: np.ndarray,
    dimension: int,
    config: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fit an outcome-free autoencoder on training profiles only."""
    set_seed(seed)
    mean, scale = standardizer_fit(train)
    train_z = standardize(train, mean, scale).astype(np.float32)
    test_z = standardize(test, mean, scale).astype(np.float32)
    order = np.random.default_rng(seed).permutation(len(train_z))
    cut = max(1, int(0.9 * len(order)))
    fit_values = train_z[order[:cut]]
    validation = train_z[order[cut:]] if cut < len(order) else train_z[order[:1]]
    model = AverageDayAutoencoder(dimension, int(config["hidden_dimension"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best = copy.deepcopy(model.state_dict())
    best_loss, stale, history = float("inf"), 0, []
    for epoch in range(int(config["epochs"])):
        model.train()
        losses = []
        for (batch,) in _loader(fit_values, int(config["batch_size"]), True, seed + epoch):
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = nn.functional.mse_loss(reconstructed, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for (batch,) in _loader(validation, int(config["batch_size"]), False, seed):
                reconstructed, _ = model(batch)
                validation_losses.append(float(nn.functional.mse_loss(reconstructed, batch)))
        valid_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "valid_loss": valid_loss})
        if valid_loss < best_loss - 1e-5:
            best, best_loss, stale = copy.deepcopy(model.state_dict()), valid_loss, 0
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    model.load_state_dict(best)
    model.eval()

    def encode(values: np.ndarray) -> np.ndarray:
        output = []
        with torch.no_grad():
            for (batch,) in _loader(values, 512, False, seed):
                output.append(model.encoder(batch).numpy())
        return np.concatenate(output)

    return encode(train_z), encode(test_z), history


def cox_ridge_fit(data: pd.DataFrame, features: np.ndarray, alpha: float) -> np.ndarray:
    """Breslow partial-likelihood Cox correction with a transported PREDICT LP offset."""
    x = np.asarray(features, float)
    time = data.time_years.to_numpy(float)
    event = data.cvd_death.to_numpy(int)
    offset = data.lp_predict.to_numpy(float)
    order = np.argsort(time, kind="stable")
    x, time, event, offset = x[order], time[order], event[order], offset[order]
    _, first, group = np.unique(time, return_index=True, return_inverse=True)
    deaths = np.bincount(group, weights=event)
    event_groups = np.flatnonzero(deaths > 0)
    risk_rows = first[event_groups]
    death_counts = deaths[event_groups]
    event_x = (x * event[:, None]).sum(axis=0)
    sample_count = float(len(event))

    def objective(beta: np.ndarray):
        eta = offset + x @ beta
        shift = float(eta.max())
        weights = np.exp(eta - shift)
        risk_sum = np.cumsum(weights[::-1])[::-1]
        risk_x = np.cumsum((weights[:, None] * x)[::-1], axis=0)[::-1]
        log_likelihood = (event * eta).sum() - np.sum(
            death_counts * (np.log(risk_sum[risk_rows]) + shift)
        )
        gradient = event_x - np.sum(
            death_counts[:, None] * risk_x[risk_rows] / risk_sum[risk_rows, None], axis=0
        )
        value = -log_likelihood / sample_count + 0.5 * alpha * float(beta @ beta)
        derivative = -gradient / sample_count + alpha * beta
        return value, derivative

    result = minimize(
        objective,
        np.zeros(x.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"Ridge Cox optimization failed: {result.message}")
    return np.asarray(result.x)


def fit_head(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_features: np.ndarray,
    test_features: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean, scale = standardizer_fit(train_features)
    train_z = standardize(train_features, mean, scale)
    test_z = standardize(test_features, mean, scale)
    beta = cox_ridge_fit(train_data, train_z, alpha)
    train_lp = train_data.lp_predict.to_numpy(float) + train_z @ beta
    test_lp = test_data.lp_predict.to_numpy(float) + test_z @ beta
    return train_lp, test_lp, {"feature_mean": mean, "feature_scale": scale, "beta": beta}


def _fixed_features(data: pd.DataFrame, model_id: str) -> np.ndarray:
    columns = {
        "M1": ["volume_mean_cpm"],
        "M2": ["mvpa_min_day"],
        "M3": ["M10", "L5", "RA"],
    }[model_id]
    return data[columns].to_numpy(float)


def representation_features(
    model_id: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_profiles: np.ndarray,
    test_profiles: np.ndarray,
    dimension: int,
    config: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    if model_id in {"M1", "M2", "M3"}:
        return _fixed_features(train_data, model_id), _fixed_features(test_data, model_id), []
    if model_id == "M4":
        train_embedding, test_embedding = pca_transform(train_profiles, test_profiles, dimension)
        return train_embedding, test_embedding, []
    if model_id in {"M5", "M6"}:
        train_embedding, test_embedding, history = ae_transform(
            train_profiles, test_profiles, dimension, config["ae"], seed
        )
        if model_id == "M6":
            train_embedding = np.column_stack([_fixed_features(train_data, "M3"), train_embedding])
            test_embedding = np.column_stack([_fixed_features(test_data, "M3"), test_embedding])
        return train_embedding, test_embedding, history
    raise ValueError(f"No representation implementation for {model_id}")


@dataclass
class SelectedConfiguration:
    dimension: int
    alpha: float
    mean_c_index: float
    se_c_index: float


def select_configuration(
    train_data: pd.DataFrame,
    train_profiles: np.ndarray,
    model_id: str,
    config: dict,
    seed: int,
) -> tuple[SelectedConfiguration, list[dict[str, float]]]:
    """Strict inner-fold selection; representation learning stays inside each fold."""
    dimensions = config["dimensions"] if model_id in {"M4", "M5", "M6"} else [0]
    alphas = config["ridge_alphas"]
    inner = stratified_folds(train_data, int(config["inner_folds"]), seed)
    all_indices = np.arange(len(train_data))
    scores = {(int(dimension), float(alpha)): [] for dimension in dimensions for alpha in alphas}
    for inner_number, validation_indices in enumerate(inner, start=1):
        fitting_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
        fit_data = train_data.iloc[fitting_indices].reset_index(drop=True)
        valid_data = train_data.iloc[validation_indices].reset_index(drop=True)
        for dimension in dimensions:
            fit_features, valid_features, _ = representation_features(
                model_id,
                fit_data,
                valid_data,
                train_profiles[fitting_indices],
                train_profiles[validation_indices],
                int(dimension),
                config,
                seed + inner_number * 100 + int(dimension),
            )
            for alpha in alphas:
                _, valid_lp, _ = fit_head(
                    fit_data, valid_data, fit_features, valid_features, float(alpha)
                )
                scores[(int(dimension), float(alpha))].append(c_index(valid_data, valid_lp))
    records = []
    for (dimension, alpha), values in scores.items():
        array = np.asarray(values, float)
        records.append(
            {
                "dimension": dimension,
                "alpha": alpha,
                "mean_c_index": float(array.mean()),
                "se_c_index": float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0,
            }
        )
    best = max(records, key=lambda row: row["mean_c_index"])
    threshold = best["mean_c_index"] - best["se_c_index"]
    eligible = [row for row in records if row["mean_c_index"] >= threshold]
    smallest_dimension = min(row["dimension"] for row in eligible)
    selected = max(
        [row for row in eligible if row["dimension"] == smallest_dimension],
        key=lambda row: row["alpha"],
    )
    return SelectedConfiguration(**selected), records


def fit_selected_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_profiles: np.ndarray,
    test_profiles: np.ndarray,
    model_id: str,
    config: dict,
    seed: int,
) -> dict:
    if model_id == "M0":
        return {
            "train_lp": train_data.lp_predict.to_numpy(float),
            "test_lp": test_data.lp_predict.to_numpy(float),
            "selected": SelectedConfiguration(0, np.nan, np.nan, np.nan),
            "tuning": [],
            "train_features": np.empty((len(train_data), 0)),
            "test_features": np.empty((len(test_data), 0)),
            "history": [],
            "head": {},
        }
    selected, tuning = select_configuration(train_data, train_profiles, model_id, config, seed)
    train_features, test_features, history = representation_features(
        model_id,
        train_data,
        test_data,
        train_profiles,
        test_profiles,
        selected.dimension,
        config,
        seed + 5000,
    )
    train_lp, test_lp, head = fit_head(
        train_data, test_data, train_features, test_features, selected.alpha
    )
    return {
        "train_lp": train_lp,
        "test_lp": test_lp,
        "selected": selected,
        "tuning": tuning,
        "train_features": train_features,
        "test_features": test_features,
        "history": history,
        "head": head,
    }


def breslow_cumulative_hazard(
    train_data: pd.DataFrame, train_lp: np.ndarray, horizon: float
) -> tuple[float, float]:
    center = float(np.mean(train_lp))
    weights = np.exp(np.asarray(train_lp) - center)
    time = train_data.time_years.to_numpy(float)
    event = train_data.cvd_death.to_numpy(int)
    cumulative = 0.0
    for point in np.unique(time[(event == 1) & (time <= horizon)]):
        cumulative += ((time == point) & (event == 1)).sum() / weights[time >= point].sum()
    return float(cumulative), center


def absolute_risk_from_training(
    train_data: pd.DataFrame,
    train_lp: np.ndarray,
    test_lp: np.ndarray,
    horizon: float,
) -> np.ndarray:
    cumulative, center = breslow_cumulative_hazard(train_data, train_lp, horizon)
    return 1 - np.exp(-cumulative * np.exp(np.asarray(test_lp) - center))


def software_versions() -> dict[str, str]:
    import scipy
    import statsmodels

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "torch": torch.__version__,
    }
