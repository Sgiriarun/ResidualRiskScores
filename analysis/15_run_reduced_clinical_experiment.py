"""BMI-focused wearable substitution experiment.

The experiment asks two separate questions:
1. How much measured BMI can fold-safe activity representations recover?
2. Can those representations restore CVD-risk performance when the published
   PREDICT equation without BMI is used?

All wearable heads, BMI readouts, and baseline hazards are fitted inside the
existing outer training folds. The locked contrastive representation is reused
from the fold-safe AI benchmark cache.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import point_clinical_metrics
from models import (
    CACHE,
    DATA,
    FIGURES,
    HERE,
    PREDICTIONS,
    TABLES,
    absolute_risk_from_training,
    c_index,
    fit_head,
    load_config,
)


CIRCADIAN_COLUMNS = ["M10", "L5", "RA"]

METHODS = {
    "FULL_BMI": "Full PREDICT-BMI",
    "NOBMI": "PREDICT without BMI",
    "NOBMI_VOL": "+ volume",
    "NOBMI_MVPA": "+ MVPA",
    "NOBMI_CIRC": "+ circadian",
    "NOBMI_CON8": "+ contrastive-8",
    "NOBMI_BOTH": "+ circadian + contrastive-8",
}

READOUTS = {
    "VOL": "Volume",
    "MVPA": "MVPA",
    "CIRC": "Circadian",
    "CON8": "Contrastive-8",
    "BOTH": "Circadian + contrastive-8",
}

COLORS = {
    "FULL_BMI": "#222222",
    "NOBMI": "#7A8594",
    "NOBMI_VOL": "#56B4E9",
    "NOBMI_MVPA": "#0072B2",
    "NOBMI_CIRC": "#E69F00",
    "NOBMI_CON8": "#D55E00",
    "NOBMI_BOTH": "#009E73",
}


def compute_no_bmi_lp(row: pd.Series) -> float:
    """Published PREDICT NoPriorCVDRisk linear predictor without BMI."""
    sex = row["sex"]
    age = max(30.0, min(79.0, float(row["age"])))
    nzdep = float(row["nzdep"])
    maori = int(row.get("maori", 0))
    pacific = int(row.get("pacific", 0))
    indian = int(row.get("indian", 0))
    asian = int(row.get("asian", 0))
    current = int(row.get("cur_smoker", 0))
    former = 0 if current else int(row.get("ex_smoker", 0))
    diabetes = int(row.get("diabetes", 0))
    af = int(row.get("af", 0))
    family = int(row.get("familyhx", 0))
    bpl = int(row.get("bpl", 0))
    lld = int(row.get("lld", 0))
    athrombi = int(row.get("athrombi", 0))
    sbp = float(row["sbp"]) if pd.notna(row["sbp"]) else 128.0
    tchdl = float(row["tchdl"]) if pd.notna(row["tchdl"]) else 3.72

    if sex == "Female":
        ac = age - 56.13665
        nc = nzdep - 2.990826
        sc = sbp - 129.0173
        tc = tchdl - 3.726268
        return (
            0.0756412 * ac
            + 0.3910183 * maori
            + 0.2010224 * pacific
            + 0.1183427 * indian
            - 0.28551 * asian
            + 0.087476 * former
            + 0.6226384 * current
            + 0.1080795 * nc
            + 0.5447632 * diabetes
            + 0.8927126 * af
            + 0.0445534 * family
            - 0.0593798 * lld
            + 0.1172496 * athrombi
            + 0.339925 * bpl
            + 0.0136606 * sc
            + 0.1226753 * tc
            - 0.0222549 * (ac if diabetes else 0.0)
            - 0.0004425 * ac * sc
            - 0.004313 * (sc if bpl else 0.0)
        )

    ac = age - 51.79953
    nc = nzdep - 2.972793
    sc = sbp - 129.1095
    tc = tchdl - 4.38906
    return (
        0.0675532 * ac
        + 0.2899054 * maori
        + 0.1774195 * pacific
        + 0.2902049 * indian
        - 0.3975687 * asian
        + 0.0753246 * former
        + 0.5058041 * current
        + 0.0794903 * nc
        + 0.5597023 * diabetes
        + 0.5880131 * af
        + 0.1326581 * family
        - 0.0537314 * lld
        + 0.0934141 * athrombi
        + 0.2947634 * bpl
        + 0.0163778 * sc
        + 0.1283758 * tc
        - 0.020235 * (ac if diabetes else 0.0)
        - 0.0004184 * ac * sc
        - 0.0053077 * (sc if bpl else 0.0)
    )


def load_analysis(smoke: bool):
    data = pd.read_parquet(DATA / "cohort.parquet")
    suffix = "smoke" if smoke else "full"
    if smoke:
        config = load_config(True)
        rng = np.random.default_rng(919)
        events = np.flatnonzero(data.cvd_death.to_numpy(int) == 1)
        nonevents = np.flatnonzero(data.cvd_death.to_numpy(int) == 0)
        selected = np.sort(
            np.concatenate(
                [
                    events,
                    rng.choice(
                        nonevents,
                        int(config["smoke"]["non_events"]),
                        replace=False,
                    ),
                ]
            )
        )
        data = data.iloc[selected].reset_index(drop=True)
    data = data.copy()
    data["lp_no_bmi"] = data.apply(compute_no_bmi_lp, axis=1)
    return data, suffix


def with_offset(data: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    result = data.copy()
    result["lp_predict"] = np.asarray(values, float)
    return result


def fit_bmi_readout(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    train_z = (train_x - mean) / scale
    test_z = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(train_z)), train_z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return np.column_stack([np.ones(len(test_z)), test_z]) @ beta


def fold_features(data, stored):
    train_indices = stored["train_indices"].astype(int)
    test_indices = stored["test_indices"].astype(int)
    train = data.iloc[train_indices].reset_index(drop=True)
    test = data.iloc[test_indices].reset_index(drop=True)
    contrastive_train = stored["CON8_train_features"].astype(float)
    contrastive_test = stored["CON8_features"].astype(float)
    circadian_train = train[CIRCADIAN_COLUMNS].to_numpy(float)
    circadian_test = test[CIRCADIAN_COLUMNS].to_numpy(float)
    features = {
        "VOL": (
            train[["volume_mean_cpm"]].to_numpy(float),
            test[["volume_mean_cpm"]].to_numpy(float),
        ),
        "MVPA": (
            train[["mvpa_min_day"]].to_numpy(float),
            test[["mvpa_min_day"]].to_numpy(float),
        ),
        "CIRC": (circadian_train, circadian_test),
        "CON8": (contrastive_train, contrastive_test),
        "BOTH": (
            np.column_stack([circadian_train, contrastive_train]),
            np.column_stack([circadian_test, contrastive_test]),
        ),
    }
    return train_indices, test_indices, train, test, features


def run_fold(data, fold, suffix, config):
    path = CACHE / f"ai_representation_{suffix}_fold_{fold}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run 12_run_ai_representation_experiment.py first."
        )
    horizon = float(config["horizon_years"])
    alpha = float(config["ai_experiment"]["ridge_alpha"])
    prediction_rows, readout_rows = [], []
    with np.load(path) as stored:
        train_indices, test_indices, train, test, features = fold_features(
            data, stored
        )
        full_train_lp = train.lp_predict.to_numpy(float)
        full_test_lp = test.lp_predict.to_numpy(float)
        no_bmi_train_lp = train.lp_no_bmi.to_numpy(float)
        no_bmi_test_lp = test.lp_no_bmi.to_numpy(float)
        no_bmi_train = with_offset(train, no_bmi_train_lp)
        no_bmi_test = with_offset(test, no_bmi_test_lp)

        outputs = {
            "FULL_BMI": (
                full_test_lp,
                absolute_risk_from_training(
                    train, full_train_lp, full_test_lp, horizon
                ),
            ),
            "NOBMI": (
                no_bmi_test_lp,
                absolute_risk_from_training(
                    no_bmi_train,
                    no_bmi_train_lp,
                    no_bmi_test_lp,
                    horizon,
                ),
            ),
        }
        method_features = {
            "NOBMI_VOL": features["VOL"],
            "NOBMI_MVPA": features["MVPA"],
            "NOBMI_CIRC": features["CIRC"],
            "NOBMI_CON8": features["CON8"],
            "NOBMI_BOTH": features["BOTH"],
        }
        for method_id, (train_x, test_x) in method_features.items():
            train_lp, test_lp, _ = fit_head(
                no_bmi_train, no_bmi_test, train_x, test_x, alpha
            )
            outputs[method_id] = (
                test_lp,
                absolute_risk_from_training(
                    no_bmi_train, train_lp, test_lp, horizon
                ),
            )

        for method_id, (test_lp, risk) in outputs.items():
            for local, global_index in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "SEQN": int(data.iloc[global_index].SEQN),
                        "cycle": str(data.iloc[global_index].cycle),
                        "outer_fold": int(fold),
                        "method_id": method_id,
                        "method": METHODS[method_id],
                        "linear_predictor": float(test_lp[local]),
                        "risk_10y": float(risk[local]),
                    }
                )

        train_bmi = train.bmi.to_numpy(float)
        test_bmi = test.bmi.to_numpy(float)
        train_bmi_observed = np.isfinite(train_bmi)
        test_bmi_observed = np.isfinite(test_bmi)
        if train_bmi_observed.sum() < 2 or not test_bmi_observed.any():
            raise ValueError("BMI readout fold has insufficient measured BMI")
        for representation, (train_x, test_x) in features.items():
            predicted = fit_bmi_readout(
                train_x[train_bmi_observed],
                test_x[test_bmi_observed],
                train_bmi[train_bmi_observed],
            )
            observed_test_indices = test_indices[test_bmi_observed]
            observed_test_bmi = test_bmi[test_bmi_observed]
            for local, global_index in enumerate(observed_test_indices):
                readout_rows.append(
                    {
                        "SEQN": int(data.iloc[global_index].SEQN),
                        "outer_fold": int(fold),
                        "representation": representation,
                        "representation_label": READOUTS[representation],
                        "bmi_observed": float(observed_test_bmi[local]),
                        "bmi_predicted": float(predicted[local]),
                    }
                )
    return prediction_rows, readout_rows


def readout_summary(readouts, replicates, seed):
    order = list(READOUTS)
    wide = readouts.pivot(
        index="SEQN", columns="representation", values="bmi_predicted"
    )
    observed = (
        readouts[["SEQN", "bmi_observed"]]
        .drop_duplicates("SEQN")
        .set_index("SEQN")
        .loc[wide.index, "bmi_observed"]
        .to_numpy(float)
    )

    def metrics(y, prediction):
        denominator = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - np.sum((y - prediction) ** 2) / denominator
        return float(r2), float(np.mean(np.abs(y - prediction)))

    rng = np.random.default_rng(seed)
    rows = []
    for representation in order:
        predicted = wide[representation].to_numpy(float)
        r2, mae = metrics(observed, predicted)
        sampled_r2, sampled_mae = [], []
        for _ in range(replicates):
            indices = rng.integers(0, len(observed), len(observed))
            value_r2, value_mae = metrics(
                observed[indices], predicted[indices]
            )
            sampled_r2.append(value_r2)
            sampled_mae.append(value_mae)
        rows.append(
            {
                "representation": representation,
                "representation_label": READOUTS[representation],
                "r2": r2,
                "r2_ci_low": np.percentile(sampled_r2, 2.5),
                "r2_ci_high": np.percentile(sampled_r2, 97.5),
                "mae_bmi": mae,
                "mae_ci_low": np.percentile(sampled_mae, 2.5),
                "mae_ci_high": np.percentile(sampled_mae, 97.5),
            }
        )
    return pd.DataFrame(rows)


def discrimination_summary(data, predictions, replicates, seed):
    order = list(METHODS)
    lp = (
        predictions.pivot(
            index="SEQN", columns="method_id", values="linear_predictor"
        )
        .loc[data.SEQN, order]
    )
    point = {
        method: c_index(data, lp[method].to_numpy(float)) for method in order
    }
    sampled = {method: [] for method in order}
    delta_full = {method: [] for method in order}
    delta_no_bmi = {method: [] for method in order}
    delta_circadian = {method: [] for method in order}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[indices].reset_index(drop=True)
        values = {
            method: c_index(sample, lp[method].to_numpy(float)[indices])
            for method in order
        }
        for method in order:
            sampled[method].append(values[method])
            delta_full[method].append(values[method] - values["FULL_BMI"])
            delta_no_bmi[method].append(values[method] - values["NOBMI"])
            delta_circadian[method].append(
                values[method] - values["NOBMI_CIRC"]
            )

    loss = point["FULL_BMI"] - point["NOBMI"]
    loss_supported = np.percentile(delta_full["NOBMI"], 97.5) < 0
    rows = []
    for method in order:
        gain = point[method] - point["NOBMI"]
        rows.append(
            {
                "method_id": method,
                "method": METHODS[method],
                "c_index": point[method],
                "ci_low": np.percentile(sampled[method], 2.5),
                "ci_high": np.percentile(sampled[method], 97.5),
                "delta_vs_full_bmi": point[method] - point["FULL_BMI"],
                "delta_vs_full_bmi_low": np.percentile(
                    delta_full[method], 2.5
                ),
                "delta_vs_full_bmi_high": np.percentile(
                    delta_full[method], 97.5
                ),
                "delta_vs_no_bmi": gain,
                "delta_vs_no_bmi_low": np.percentile(
                    delta_no_bmi[method], 2.5
                ),
                "delta_vs_no_bmi_high": np.percentile(
                    delta_no_bmi[method], 97.5
                ),
                "delta_vs_circadian": point[method]
                - point["NOBMI_CIRC"],
                "delta_vs_circadian_low": np.percentile(
                    delta_circadian[method], 2.5
                ),
                "delta_vs_circadian_high": np.percentile(
                    delta_circadian[method], 97.5
                ),
                "bmi_cindex_loss": loss,
                "bmi_cindex_loss_supported": loss_supported,
                "cindex_recovery_percent": (
                    100 * gain / loss
                    if loss > 0 and loss_supported
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def clinical_summary(data, predictions, horizon):
    order = list(METHODS)
    lp = (
        predictions.pivot(
            index="SEQN", columns="method_id", values="linear_predictor"
        )
        .loc[data.SEQN]
    )
    risk = (
        predictions.pivot(index="SEQN", columns="method_id", values="risk_10y")
        .loc[data.SEQN]
    )
    values = point_clinical_metrics(
        data,
        {method: lp[method].to_numpy(float) for method in order},
        {method: risk[method].to_numpy(float) for method in order},
        horizon,
    )
    return pd.DataFrame(
        [
            {"method_id": method, "method": METHODS[method], **values[method]}
            for method in order
        ]
    )


def make_figure(results, readout, smoke):
    order = list(METHODS)
    result = results.set_index("method_id").loc[order]
    readout_order = list(READOUTS)
    readout = readout.set_index("representation").loc[readout_order]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.3))
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")

    y_readout = np.arange(len(readout_order))[::-1]
    axes[0].errorbar(
        readout.r2,
        y_readout,
        xerr=[
            readout.r2 - readout.r2_ci_low,
            readout.r2_ci_high - readout.r2,
        ],
        fmt="o",
        color="#0072B2",
        capsize=3,
    )
    axes[0].axvline(0, color="#555555", ls="--", lw=1)
    axes[0].set_yticks(y_readout)
    axes[0].set_yticklabels(readout.representation_label)
    axes[0].set_xlabel("Out-of-fold BMI R2 (95% CI)")
    axes[0].set_title("A. Can activity estimate BMI?", loc="left", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.2)

    y = np.arange(len(order))[::-1]
    for position, method in enumerate(order):
        row = result.loc[method]
        axes[1].errorbar(
            row.c_index,
            y[position],
            xerr=[
                [row.c_index - row.ci_low],
                [row.ci_high - row.c_index],
            ],
            fmt="o",
            color=COLORS[method],
            capsize=3,
        )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([METHODS[method] for method in order])
    axes[1].set_xlabel("Out-of-fold C-index (95% CI)")
    axes[1].set_title("B. CVD risk ranking", loc="left", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.2)

    augmented = order[2:]
    y_gain = np.arange(len(augmented))[::-1]
    for position, method in enumerate(augmented):
        row = result.loc[method]
        point = 1000 * row.delta_vs_no_bmi
        low = 1000 * row.delta_vs_no_bmi_low
        high = 1000 * row.delta_vs_no_bmi_high
        axes[2].errorbar(
            point,
            y_gain[position],
            xerr=[[point - low], [high - point]],
            fmt="o",
            color=COLORS[method],
            capsize=3,
        )
    axes[2].axvline(0, color="#555555", ls="--", lw=1)
    axes[2].set_yticks(y_gain)
    axes[2].set_yticklabels([METHODS[method] for method in augmented])
    axes[2].set_xlabel("Paired gain vs no-BMI PREDICT (x1000)")
    axes[2].set_title("C. Added value without BMI", loc="left", fontweight="bold")
    axes[2].grid(axis="x", alpha=0.2)

    fig.suptitle(
        "Does average-day accelerometry add value when BMI is unavailable?",
        x=0.04,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.01,
        "Published PREDICT equations with and without BMI; every wearable representation and correction is outer-fold safe.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.91])
    suffix = "_smoke" if smoke else ""
    fig.savefig(
        FIGURES / f"fig13_reduced_clinical_recovery{suffix}.png",
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURES / f"fig13_reduced_clinical_recovery{suffix}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def write_summary(results, clinical, readout, data, smoke, config):
    result = results.set_index("method_id")
    clinical = clinical.set_index("method_id")
    best_id = result.loc[list(METHODS)[2:]].c_index.idxmax()
    best = result.loc[best_id]
    bmi_change = result.loc["NOBMI"]
    best_readout_id = readout.set_index("representation").r2.idxmax()
    best_readout = readout.set_index("representation").loc[best_readout_id]
    combined = result.loc["NOBMI_BOTH"]
    loss = bmi_change.bmi_cindex_loss
    if bool(bmi_change.bmi_cindex_loss_supported):
        recovery = (
            f"The best wearable model recovered {best.cindex_recovery_percent:.1f}% "
            "of the C-index loss."
        )
    else:
        recovery = (
            "A BMI-related loss was not established because the paired confidence "
            "interval included zero; therefore, a recovery percentage is not "
            "reported."
        )
    endpoint_label = config.get("endpoint", {}).get("label", "CVD deaths")
    endpoint_description = config.get("endpoint", {}).get("description", "mortality endpoint")
    text = f"""# BMI-focused wearable substitution experiment

This analysis used **{len(data):,} participants** and **{int(data.cvd_death.sum())}
{endpoint_label}**. It compared the published PREDICT equations with and
without BMI using the same outer folds, then added fold-safe average-day activity
representations to the no-BMI equation.

## Can activity estimate BMI?

The strongest activity readout was **{READOUTS[best_readout_id]}**, with
cross-validated BMI R2 **{best_readout.r2:.3f}** (95% CI
{best_readout.r2_ci_low:.3f} to {best_readout.r2_ci_high:.3f}) and MAE
**{best_readout.mae_bmi:.2f} kg/m2**. This measures statistical recoverability;
it does not mean that an accelerometer directly measures body mass.

## CVD-risk results

- Full PREDICT-BMI: C-index **{result.loc['FULL_BMI', 'c_index']:.4f}**.
- Published PREDICT without BMI: C-index **{result.loc['NOBMI', 'c_index']:.4f}**;
  paired change versus the BMI equation **{bmi_change.delta_vs_full_bmi:+.4f}**
  (95% CI {bmi_change.delta_vs_full_bmi_low:+.4f} to
  {bmi_change.delta_vs_full_bmi_high:+.4f}).
- Best wearable augmentation: **{METHODS[best_id]}**, C-index
  **{best.c_index:.4f}**; paired gain versus no-BMI PREDICT
  **{best.delta_vs_no_bmi:+.4f}** (95% CI {best.delta_vs_no_bmi_low:+.4f} to
  {best.delta_vs_no_bmi_high:+.4f}). {recovery}
- Contrastive information beyond circadian: **{combined.delta_vs_circadian:+.4f}**
  (95% CI {combined.delta_vs_circadian_low:+.4f} to
  {combined.delta_vs_circadian_high:+.4f}).

The best model's ten-year IPCW Brier score was
**{clinical.loc[best_id, 'brier_10y']:.5f}**, compared with
**{clinical.loc['FULL_BMI', 'brier_10y']:.5f}** for full PREDICT-BMI and
**{clinical.loc['NOBMI', 'brier_10y']:.5f}** for PREDICT without BMI.

## Interpretation

Accelerometry can be described as compensating for unavailable BMI only if the
no-BMI equation loses predictive performance and adding activity restores that
loss. That prerequisite was not demonstrated here. BMI readout R2 answers a
different question: whether activity contains an obesity-related behavioural
phenotype; the low R2 shows that it contains little direct BMI information.
Therefore, the wearable result is best interpreted as independent activity-based
augmentation of a no-BMI risk score, not BMI replacement. The NHANES
{endpoint_description} also differs from the original PREDICT outcome, so this
remains a transported proof-of-concept analysis rather than a deployable no-BMI
risk model.
"""
    target = HERE / (
        "REDUCED_CLINICAL_RESULTS_SMOKE.md"
        if smoke
        else "REDUCED_CLINICAL_RESULTS.md"
    )
    target.write_text(text)


def verify(data, predictions, readouts):
    expected = len(METHODS)
    if not (predictions.groupby("SEQN").method_id.nunique() == expected).all():
        raise AssertionError("Every participant must have every risk prediction")
    if predictions.duplicated(["SEQN", "method_id"]).any():
        raise AssertionError("Duplicate risk predictions")
    readout_counts = readouts.groupby("SEQN").representation.nunique()
    if not (readout_counts == len(READOUTS)).all():
        raise AssertionError("Every participant must have every BMI readout")
    expected_readout = set(data.loc[data.bmi.notna(), "SEQN"].astype(int))
    if set(readout_counts.index.astype(int)) != expected_readout:
        raise AssertionError("BMI readout participants do not match measured BMI")
    full = (
        predictions[predictions.method_id == "FULL_BMI"]
        .set_index("SEQN")
        .loc[data.SEQN]
    )
    if not np.allclose(full.linear_predictor, data.lp_predict):
        raise AssertionError("Full PREDICT-BMI LP was not reproduced")
    if not np.isfinite(data.lp_no_bmi).all():
        raise AssertionError("Non-finite no-BMI PREDICT LP")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.smoke)
    data, suffix = load_analysis(args.smoke)
    prediction_rows, readout_rows = [], []
    for fold in range(1, int(config["outer_folds"]) + 1):
        fold_predictions, fold_readouts = run_fold(
            data, fold, suffix, config
        )
        prediction_rows.extend(fold_predictions)
        readout_rows.extend(fold_readouts)
    predictions = (
        pd.DataFrame(prediction_rows)
        .sort_values(["SEQN", "method_id"])
        .reset_index(drop=True)
    )
    readouts = (
        pd.DataFrame(readout_rows)
        .sort_values(["SEQN", "representation"])
        .reset_index(drop=True)
    )
    verify(data, predictions, readouts)

    output_suffix = "_smoke" if args.smoke else ""
    predictions.to_parquet(
        PREDICTIONS / f"reduced_clinical_predictions{output_suffix}.parquet",
        index=False,
    )
    readouts.to_parquet(
        PREDICTIONS / f"bmi_readout_predictions{output_suffix}.parquet",
        index=False,
    )
    results = discrimination_summary(
        data,
        predictions,
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + 1501,
    )
    bmi_readout = readout_summary(
        readouts,
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + 1502,
    )
    clinical = clinical_summary(
        data, predictions, float(config["horizon_years"])
    )
    results.to_csv(
        TABLES / f"reduced_clinical_results{output_suffix}.csv", index=False
    )
    clinical.to_csv(
        TABLES / f"reduced_clinical_clinical{output_suffix}.csv", index=False
    )
    bmi_readout.to_csv(
        TABLES / f"bmi_readout_results{output_suffix}.csv", index=False
    )
    make_figure(results, bmi_readout, args.smoke)
    write_summary(results, clinical, bmi_readout, data, args.smoke, config)
    print(
        results[
            [
                "method",
                "c_index",
                "delta_vs_no_bmi",
                "delta_vs_circadian",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nBMI readout:")
    print(
        bmi_readout[["representation_label", "r2", "mae_bmi"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
