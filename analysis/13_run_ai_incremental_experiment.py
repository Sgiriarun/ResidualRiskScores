"""Test whether fixed 8-D AI representations add beyond circadian features."""

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


METHODS = {
    "B0": "Transported PREDICT LP",
    "B1": "PREDICT LP + circadian",
    "CP8": "PREDICT LP + circadian + PCA-8",
    "CAE8": "PREDICT LP + circadian + reconstruction AE-8",
    "CMAE8": "PREDICT LP + circadian + masked AE-8",
    "CCON8": "PREDICT LP + circadian + contrastive-8",
    "CRAE8": "PREDICT LP + circadian + risk-aware masked AE-8",
}
REPRESENTATIONS = {
    "CP8": "P8",
    "CAE8": "AE8",
    "CMAE8": "MAE8",
    "CCON8": "CON8",
    "CRAE8": "RAE8",
}
COLORS = {
    "B0": "#7A8594",
    "B1": "#E69F00",
    "CP8": "#7A5AB5",
    "CAE8": "#009E73",
    "CMAE8": "#2B8CBE",
    "CCON8": "#D55E00",
    "CRAE8": "#C44E52",
}


def load_inputs(smoke: bool):
    data = pd.read_parquet(DATA / "cohort.parquet")
    suffix = "smoke" if smoke else "full"
    if smoke:
        config = load_config(True)
        rng = np.random.default_rng(919)
        events = np.flatnonzero(data.cvd_death.to_numpy(int) == 1)
        nonevents = np.flatnonzero(data.cvd_death.to_numpy(int) == 0)
        selected = np.sort(
            np.concatenate(
                [events, rng.choice(nonevents, int(config["smoke"]["non_events"]), replace=False)]
            )
        )
        data = data.iloc[selected].reset_index(drop=True)
    return data, suffix


def fit_fold(data: pd.DataFrame, fold: int, suffix: str, alpha: float, horizon: float):
    cache_path = CACHE / f"ai_representation_{suffix}_fold_{fold}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}. Run 12_run_ai_representation_experiment.py first."
        )
    with np.load(cache_path) as stored:
        train_indices = stored["train_indices"].astype(int)
        test_indices = stored["test_indices"].astype(int)
        train_data = data.iloc[train_indices].reset_index(drop=True)
        test_data = data.iloc[test_indices].reset_index(drop=True)
        rows = []

        for method_id in ("B0", "B1"):
            for local, global_index in enumerate(test_indices):
                rows.append(
                    {
                        "SEQN": int(data.iloc[global_index].SEQN),
                        "outer_fold": fold,
                        "method_id": method_id,
                        "linear_predictor": float(stored[f"{method_id}_lp"][local]),
                        "risk_10y": float(stored[f"{method_id}_risk"][local]),
                    }
                )

        circadian_columns = ["M10", "L5", "RA"]
        train_circadian = train_data[circadian_columns].to_numpy(float)
        test_circadian = test_data[circadian_columns].to_numpy(float)
        for method_id, representation_id in REPRESENTATIONS.items():
            train_features = np.column_stack(
                [train_circadian, stored[f"{representation_id}_train_features"]]
            )
            test_features = np.column_stack(
                [test_circadian, stored[f"{representation_id}_features"]]
            )
            train_lp, test_lp, _ = fit_head(
                train_data, test_data, train_features, test_features, alpha
            )
            risk = absolute_risk_from_training(
                train_data, train_lp, test_lp, horizon
            )
            for local, global_index in enumerate(test_indices):
                rows.append(
                    {
                        "SEQN": int(data.iloc[global_index].SEQN),
                        "outer_fold": fold,
                        "method_id": method_id,
                        "linear_predictor": float(test_lp[local]),
                        "risk_10y": float(risk[local]),
                    }
                )
    return pd.DataFrame(rows)


def discrimination_summary(data, predictions, replicates, seed):
    order = list(METHODS)
    lp = (
        predictions.pivot(index="SEQN", columns="method_id", values="linear_predictor")
        .loc[data.SEQN, order]
    )
    point = {method: c_index(data, lp[method].to_numpy(float)) for method in order}
    sampled = {method: [] for method in order}
    deltas = {method: [] for method in order}
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
            deltas[method].append(values[method] - values["B1"])
    rows = []
    for method in order:
        low, high = np.percentile(sampled[method], [2.5, 97.5])
        delta_low, delta_high = np.percentile(deltas[method], [2.5, 97.5])
        rows.append(
            {
                "method_id": method,
                "method": METHODS[method],
                "c_index": point[method],
                "ci_low": low,
                "ci_high": high,
                "delta_vs_B1": point[method] - point["B1"],
                "delta_vs_B1_low": delta_low,
                "delta_vs_B1_high": delta_high,
                "bootstrap_probability_delta_positive": float(
                    np.mean(np.asarray(deltas[method]) > 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def clinical_summary(data, predictions, horizon):
    order = list(METHODS)
    lp = predictions.pivot(index="SEQN", columns="method_id", values="linear_predictor").loc[data.SEQN]
    risk = predictions.pivot(index="SEQN", columns="method_id", values="risk_10y").loc[data.SEQN]
    values = point_clinical_metrics(
        data,
        {method: lp[method].to_numpy(float) for method in order},
        {method: risk[method].to_numpy(float) for method in order},
        horizon,
    )
    return pd.DataFrame(
        [{"method_id": method, "method": METHODS[method], **values[method]} for method in order]
    )


def make_figure(results, smoke):
    order = list(METHODS)
    data = results.set_index("method_id").loc[order]
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1), gridspec_kw={"width_ratios": [1, 1.05]})
    for position, method in enumerate(order):
        row = data.loc[method]
        axes[0].errorbar(
            row.c_index,
            y[position],
            xerr=[[row.c_index - row.ci_low], [row.ci_high - row.c_index]],
            fmt="o",
            color=COLORS[method],
            capsize=3,
        )
        axes[1].errorbar(
            row.delta_vs_B1,
            y[position],
            xerr=[
                [row.delta_vs_B1 - row.delta_vs_B1_low],
                [row.delta_vs_B1_high - row.delta_vs_B1],
            ],
            fmt="o",
            color=COLORS[method],
            capsize=3,
        )
    labels = [
        METHODS[method]
        .replace("PREDICT LP + circadian + ", "+ ")
        .replace("PREDICT LP + circadian", "Circadian")
        for method in order
    ]
    labels[0] = "PREDICT"
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_xlabel("Out-of-fold C-index (95% CI)")
    axes[0].set_title("A. Circadian plus each representation", loc="left", fontweight="bold")
    axes[1].axvline(0, color="#555555", ls="--", lw=1)
    axes[1].set_xlabel("Paired change vs circadian alone (95% CI)")
    axes[1].set_title("B. Incremental AI information", loc="left", fontweight="bold")
    fig.suptitle(
        "Does an average-day AI representation add beyond known rhythm features?",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.01,
        "Exploratory fixed configuration: 8 dimensions, ridge alpha=0.1; identical outer folds and fold-safe embeddings.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    suffix = "_smoke" if smoke else ""
    fig.savefig(FIGURES / f"fig11_ai_incremental_beyond_circadian{suffix}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGURES / f"fig11_ai_incremental_beyond_circadian{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(results, clinical, data, smoke, config):
    indexed = results.set_index("method_id")
    clinical = clinical.set_index("method_id")
    candidates = indexed.loc[list(REPRESENTATIONS)]
    best = candidates.c_index.idxmax()
    row = indexed.loc[best]
    endpoint_label = config.get("endpoint", {}).get("label", "CVD deaths")
    if row.delta_vs_B1_low > 0:
        inference = "The paired 95% confidence interval excluded zero."
    elif row.delta_vs_B1_high < 0:
        inference = "The paired 95% confidence interval was below zero."
    else:
        inference = "The paired 95% confidence interval included zero."
    text = f"""# AI representations beyond circadian features

This exploratory incremental experiment used **{len(data):,} participants** and
**{int(data.cvd_death.sum())} {endpoint_label}**. It reused the outer-fold-safe
8-dimensional representations from the objective benchmark and refitted each
Cox correction using M10, L5, RA and the representation together. Ridge alpha
was fixed at 0.1 for every model.

## Result

The highest combined-model point estimate was **{METHODS[best]}**, with C-index
**{row.c_index:.4f}**. Its paired change versus PREDICT LP + circadian alone was
**{row.delta_vs_B1:+.4f}** (95% CI {row.delta_vs_B1_low:+.4f} to
{row.delta_vs_B1_high:+.4f}). {inference}

Its ten-year IPCW Brier score was **{clinical.loc[best, 'brier_10y']:.5f}**, compared
with **{clinical.loc['B1', 'brier_10y']:.5f}** for circadian alone. Its net benefit
at 10% was **{clinical.loc[best, 'net_benefit_10pct']:.5f}**, compared with
**{clinical.loc['B1', 'net_benefit_10pct']:.5f}**. These clinical metrics are point
estimates and the risk thresholds remain exploratory.

## Interpretation

Unlike the alternative-model benchmark, this experiment directly asks whether
the learned profile contains prognostic information after known circadian
features are already included. Because objectives and the eight-dimensional
configuration were examined after looking at earlier results, this remains an
exploratory analysis and requires seed-stability and external validation before
confirmatory claims.
"""
    target = HERE / ("AI_INCREMENTAL_RESULTS_SMOKE.md" if smoke else "AI_INCREMENTAL_RESULTS.md")
    target.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.smoke)
    data, suffix = load_inputs(args.smoke)
    folds = sorted(
        pd.read_parquet(
            PREDICTIONS / f"ai_representation_predictions{'_smoke' if args.smoke else ''}.parquet"
        ).outer_fold.unique()
    )
    alpha = float(config["ai_experiment"]["ridge_alpha"])
    horizon = float(config["horizon_years"])
    parts = [fit_fold(data, int(fold), suffix, alpha, horizon) for fold in folds]
    predictions = pd.concat(parts, ignore_index=True).sort_values(["SEQN", "method_id"])
    predictions["method"] = predictions.method_id.map(METHODS)
    output_suffix = "_smoke" if args.smoke else ""
    predictions.to_parquet(
        PREDICTIONS / f"ai_incremental_predictions{output_suffix}.parquet", index=False
    )
    results = discrimination_summary(
        data,
        predictions,
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + 1301,
    )
    clinical = clinical_summary(data, predictions, horizon)
    results.to_csv(TABLES / f"ai_incremental_results{output_suffix}.csv", index=False)
    clinical.to_csv(TABLES / f"ai_incremental_clinical{output_suffix}.csv", index=False)
    make_figure(results, args.smoke)
    write_summary(results, clinical, data, args.smoke, config)
    print(results.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
