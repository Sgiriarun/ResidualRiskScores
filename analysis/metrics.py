"""Out-of-fold discrimination, calibration, Brier, and decision-curve metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

from models import MODEL_IDS, c_index


def km_event_probability(data: pd.DataFrame, horizon: float, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(data), dtype=bool)
    time = data.time_years.to_numpy(float)[mask]
    event = data.cvd_death.to_numpy(int)[mask]
    survival = 1.0
    for point in np.unique(time[(event == 1) & (time <= horizon)]):
        at_risk = (time >= point).sum()
        if at_risk:
            survival *= 1 - ((time == point) & (event == 1)).sum() / at_risk
    return float(1 - survival)


def censoring_survival(data: pd.DataFrame, query: np.ndarray, horizon: float) -> np.ndarray:
    time = data.time_years.to_numpy(float)
    censor = (data.cvd_death.to_numpy(int) == 0).astype(int)
    before, survival = {}, 1.0
    for point in np.unique(time):
        before[point] = survival
        at_risk = (time >= point).sum()
        if at_risk:
            survival *= 1 - ((time == point) & (censor == 1)).sum() / at_risk
    keys = np.asarray(sorted(before))
    values = np.asarray([before[key] for key in keys])
    locations = np.clip(
        np.searchsorted(keys, np.minimum(query, horizon), side="right") - 1,
        0,
        len(keys) - 1,
    )
    return np.maximum(values[locations], 1e-6)


def ipcw_brier(data: pd.DataFrame, predicted: np.ndarray, horizon: float) -> float:
    time = data.time_years.to_numpy(float)
    event = data.cvd_death.to_numpy(int)
    by_horizon = (time <= horizon) & (event == 1)
    survived = time > horizon
    event_error = (
        (1 - predicted[by_horizon]) ** 2
        / censoring_survival(data, time[by_horizon], horizon)
    ).sum()
    survival_error = (
        predicted[survived] ** 2
        / censoring_survival(data, np.full(survived.sum(), horizon), horizon)
    ).sum()
    return float((event_error + survival_error) / len(data))


def calibration_slope(data: pd.DataFrame, linear_predictor: np.ndarray) -> float:
    try:
        result = PHReg(
            endog=data.time_years.to_numpy(float),
            exog=np.asarray(linear_predictor, float)[:, None],
            status=data.cvd_death.to_numpy(int),
        ).fit(disp=False)
        return float(result.params[0])
    except Exception:
        return float("nan")


def net_benefit(data: pd.DataFrame, predicted: np.ndarray, threshold: float, horizon: float) -> float:
    treated = np.asarray(predicted) >= threshold
    if not treated.any():
        return 0.0
    event_probability = km_event_probability(data, horizon, treated)
    penalty = threshold / (1 - threshold)
    return float((event_probability - (1 - event_probability) * penalty) * treated.mean())


def decision_curve(
    data: pd.DataFrame,
    risks: dict[str, np.ndarray],
    horizon: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.01, 0.201, 0.01)
    prevalence = km_event_probability(data, horizon)
    rows = []
    for threshold in thresholds:
        penalty = threshold / (1 - threshold)
        row = {
            "threshold": float(threshold),
            "treat_none": 0.0,
            "treat_all": float(prevalence - (1 - prevalence) * penalty),
        }
        row.update(
            {
                model_id: net_benefit(data, predicted, float(threshold), horizon)
                for model_id, predicted in risks.items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(
    data: pd.DataFrame,
    risks: dict[str, np.ndarray],
    horizon: float,
    groups: int = 10,
) -> pd.DataFrame:
    rows = []
    for model_id, predicted in risks.items():
        group = pd.qcut(pd.Series(predicted).rank(method="first"), groups, labels=False).to_numpy()
        for number in range(groups):
            mask = group == number
            rows.append(
                {
                    "model_id": model_id,
                    "risk_decile": number + 1,
                    "participants": int(mask.sum()),
                    "mean_predicted_10y_risk": float(np.mean(predicted[mask])),
                    "observed_10y_km_risk": km_event_probability(data, horizon, mask),
                }
            )
    return pd.DataFrame(rows)


def point_clinical_metrics(
    data: pd.DataFrame,
    linear_predictors: dict[str, np.ndarray],
    risks: dict[str, np.ndarray],
    horizon: float,
) -> dict[str, dict[str, float]]:
    observed = km_event_probability(data, horizon)
    output = {}
    for model_id in linear_predictors:
        predicted = risks[model_id]
        output[model_id] = {
            "calibration_slope": calibration_slope(data, linear_predictors[model_id]),
            "mean_predicted_10y_risk": float(np.mean(predicted)),
            "observed_10y_km_risk": observed,
            "calibration_in_large_difference": float(np.mean(predicted) - observed),
            "brier_10y": ipcw_brier(data, predicted, horizon),
            "net_benefit_5pct": net_benefit(data, predicted, 0.05, horizon),
            "net_benefit_10pct": net_benefit(data, predicted, 0.10, horizon),
            "net_benefit_15pct": net_benefit(data, predicted, 0.15, horizon),
        }
    return output


def bootstrap_discrimination(
    data: pd.DataFrame,
    linear_predictors: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    point = {model_id: c_index(data, values) for model_id, values in linear_predictors.items()}
    sampled = {model_id: [] for model_id in linear_predictors}
    delta_base = {model_id: [] for model_id in linear_predictors}
    delta_circadian = {model_id: [] for model_id in linear_predictors}
    for _ in range(replicates):
        indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[indices].reset_index(drop=True)
        values = {
            model_id: c_index(sample, predictor[indices])
            for model_id, predictor in linear_predictors.items()
        }
        for model_id, value in values.items():
            sampled[model_id].append(value)
            delta_base[model_id].append(value - values["M0"])
            delta_circadian[model_id].append(value - values["M3"])
    rows = []
    for model_id in MODEL_IDS:
        low, high = np.percentile(sampled[model_id], [2.5, 97.5])
        base_low, base_high = np.percentile(delta_base[model_id], [2.5, 97.5])
        circ_low, circ_high = np.percentile(delta_circadian[model_id], [2.5, 97.5])
        rows.append(
            {
                "model_id": model_id,
                "c_index": point[model_id],
                "ci_low": low,
                "ci_high": high,
                "delta_vs_M0": point[model_id] - point["M0"],
                "delta_vs_M0_low": base_low,
                "delta_vs_M0_high": base_high,
                "probability_delta_vs_M0_positive": float(np.mean(np.asarray(delta_base[model_id]) > 0)),
                "delta_vs_M3": point[model_id] - point["M3"],
                "delta_vs_M3_low": circ_low,
                "delta_vs_M3_high": circ_high,
                "probability_delta_vs_M3_positive": float(np.mean(np.asarray(delta_circadian[model_id]) > 0)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_clinical_metrics(
    data: pd.DataFrame,
    linear_predictors: dict[str, np.ndarray],
    risks: dict[str, np.ndarray],
    horizon: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    point = point_clinical_metrics(data, linear_predictors, risks, horizon)
    distributions = {
        model_id: {metric: [] for metric in values}
        for model_id, values in point.items()
    }
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[indices].reset_index(drop=True)
        sampled_lp = {model_id: values[indices] for model_id, values in linear_predictors.items()}
        sampled_risk = {model_id: values[indices] for model_id, values in risks.items()}
        metrics = point_clinical_metrics(sample, sampled_lp, sampled_risk, horizon)
        for model_id, values in metrics.items():
            for metric, value in values.items():
                distributions[model_id][metric].append(value)
    rows = []
    for model_id in MODEL_IDS:
        row = {"model_id": model_id}
        for metric, value in point[model_id].items():
            values = np.asarray(distributions[model_id][metric], float)
            values = values[np.isfinite(values)]
            row[metric] = value
            row[f"{metric}_ci_low"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"{metric}_ci_high"] = float(np.percentile(values, 97.5)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aalen_johansen_cif(
    time: np.ndarray,
    event_type: np.ndarray,
    horizon: float,
    mask: np.ndarray | None = None,
) -> float:
    if mask is None:
        mask = np.ones(len(time), dtype=bool)
    time = np.asarray(time, float)[mask]
    event_type = np.asarray(event_type, int)[mask]
    survival, cif = 1.0, 0.0
    for point in np.unique(time[(event_type > 0) & (time <= horizon)]):
        at_risk = (time >= point).sum()
        target = ((time == point) & (event_type == 1)).sum()
        all_events = ((time == point) & (event_type > 0)).sum()
        if at_risk:
            cif += survival * target / at_risk
            survival *= 1 - all_events / at_risk
    return float(cif)

