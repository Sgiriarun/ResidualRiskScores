"""Simple fold-safe benchmark of average-day AI representation objectives."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_representation_models import (
    risk_aware_finetune,
    train_autoencoder,
    train_contrastive,
)
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
    pca_transform,
    stratified_folds,
)


METHODS = {
    "B0": "Transported PREDICT LP",
    "B1": "PREDICT LP + circadian",
    "P8": "PREDICT LP + PCA-8",
    "AE8": "PREDICT LP + reconstruction AE-8",
    "MAE8": "PREDICT LP + masked AE-8",
    "CON8": "PREDICT LP + contrastive-8",
    "RAE8": "PREDICT LP + risk-aware masked AE-8",
}
REPRESENTATION_METHODS = ("P8", "AE8", "MAE8", "CON8", "RAE8")
READOUT_TARGETS = {
    "Volume": "volume_mean_cpm",
    "MVPA": "mvpa_min_day",
    "M10": "M10",
    "L5": "L5",
    "RA": "RA",
}
COLORS = {
    "B0": "#7A8594",
    "B1": "#E69F00",
    "P8": "#7A5AB5",
    "AE8": "#009E73",
    "MAE8": "#2B8CBE",
    "CON8": "#D55E00",
    "RAE8": "#C44E52",
}


def analysis_inputs(smoke: bool, config: dict):
    data = pd.read_parquet(DATA / "cohort.parquet")
    with np.load(DATA / "average_day_profiles.npz") as stored:
        profiles = stored["profile"].astype(np.float32)
    if smoke:
        rng = np.random.default_rng(919)
        events = np.flatnonzero(data.cvd_death.to_numpy(int) == 1)
        nonevents = np.flatnonzero(data.cvd_death.to_numpy(int) == 0)
        selected = np.sort(
            np.concatenate(
                [events, rng.choice(nonevents, int(config["smoke"]["non_events"]), replace=False)]
            )
        )
        data = data.iloc[selected].reset_index(drop=True)
        profiles = profiles[selected]
        folds = stratified_folds(data, int(config["outer_folds"]), int(config["seed"]) + 1200)
    else:
        assignments = pd.read_csv(TABLES / "fold_assignments.csv").set_index("SEQN")
        fold_number = assignments.loc[data.SEQN, "outer_fold"].to_numpy(int)
        folds = [
            np.flatnonzero(fold_number == number)
            for number in range(1, int(config["outer_folds"]) + 1)
        ]
    return data, profiles, folds


def head_and_risk(train_data, test_data, train_features, test_features, alpha, horizon):
    train_lp, test_lp, _ = fit_head(
        train_data,
        test_data,
        train_features,
        test_features,
        alpha,
    )
    risk = absolute_risk_from_training(train_data, train_lp, test_lp, horizon)
    return train_lp, test_lp, risk


def run_fold(data, profiles, test_indices, fold, config, smoke, force):
    suffix = "smoke" if smoke else "full"
    cache_path = CACHE / f"ai_representation_{suffix}_fold_{fold}.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path) as stored:
            if not all(f"{method}_features" in stored for method in REPRESENTATION_METHODS):
                stored.close()
            else:
                rows = []
                for method_id in METHODS:
                    for local, global_index in enumerate(stored["test_indices"].astype(int)):
                        rows.append(
                            {
                                "SEQN": int(data.iloc[global_index].SEQN),
                                "outer_fold": fold,
                                "method_id": method_id,
                                "linear_predictor": float(stored[f"{method_id}_lp"][local]),
                                "risk_10y": float(stored[f"{method_id}_risk"][local]),
                            }
                        )
                readouts = readout_rows(
                    data.iloc[stored["train_indices"].astype(int)].reset_index(drop=True),
                    data.iloc[stored["test_indices"].astype(int)].reset_index(drop=True),
                    {
                        method: (
                            stored[f"{method}_train_features"],
                            stored[f"{method}_features"],
                        )
                        for method in REPRESENTATION_METHODS
                    },
                    fold,
                )
                return pd.DataFrame(rows), pd.DataFrame(), readouts

    all_indices = np.arange(len(data))
    train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
    train_data = data.iloc[train_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    train_profiles = profiles[train_indices]
    test_profiles = profiles[test_indices]
    ai_config = dict(config["ai_experiment"])
    if smoke:
        ai_config["epochs"] = int(config["smoke"]["ai_epochs"])
        ai_config["risk_finetune_epochs"] = int(config["smoke"]["ai_risk_epochs"])
    alpha = float(ai_config["ridge_alpha"])
    horizon = float(config["horizon_years"])
    predictions = {}
    representations = {}
    histories = []

    print(
        f"AI fold {fold}: train={len(train_data):,}/{int(train_data.cvd_death.sum())} events; "
        f"test={len(test_data):,}/{int(test_data.cvd_death.sum())} events",
        flush=True,
    )

    train_lp = train_data.lp_predict.to_numpy(float)
    test_lp = test_data.lp_predict.to_numpy(float)
    predictions["B0"] = (
        test_lp,
        absolute_risk_from_training(train_data, train_lp, test_lp, horizon),
    )

    circadian_columns = ["M10", "L5", "RA"]
    _, lp, risk = head_and_risk(
        train_data,
        test_data,
        train_data[circadian_columns].to_numpy(float),
        test_data[circadian_columns].to_numpy(float),
        alpha,
        horizon,
    )
    predictions["B1"] = (lp, risk)

    print("  PCA-8", flush=True)
    pca_train, pca_test = pca_transform(
        train_profiles, test_profiles, int(ai_config["dimension"])
    )
    representations["P8"] = (pca_train, pca_test)
    _, lp, risk = head_and_risk(
        train_data, test_data, pca_train, pca_test, alpha, horizon
    )
    predictions["P8"] = (lp, risk)

    seed = int(config["seed"]) + fold * 10000 + 1200
    print("  Reconstruction AE-8", flush=True)
    ae_train, ae_test, _, _, _, history = train_autoencoder(
        train_profiles, test_profiles, ai_config, seed, masked=False
    )
    representations["AE8"] = (ae_train, ae_test)
    histories.extend({"outer_fold": fold, "method_id": "AE8", **row} for row in history)
    _, lp, risk = head_and_risk(
        train_data, test_data, ae_train, ae_test, alpha, horizon
    )
    predictions["AE8"] = (lp, risk)

    print("  Masked AE-8", flush=True)
    mae_train, mae_test, masked_model, train_z, test_z, history = train_autoencoder(
        train_profiles, test_profiles, ai_config, seed + 100, masked=True
    )
    representations["MAE8"] = (mae_train, mae_test)
    histories.extend({"outer_fold": fold, "method_id": "MAE8", **row} for row in history)
    _, lp, risk = head_and_risk(
        train_data, test_data, mae_train, mae_test, alpha, horizon
    )
    predictions["MAE8"] = (lp, risk)

    print("  Contrastive-8", flush=True)
    contrastive_train, contrastive_test, history = train_contrastive(
        train_profiles, test_profiles, ai_config, seed + 200
    )
    representations["CON8"] = (contrastive_train, contrastive_test)
    histories.extend({"outer_fold": fold, "method_id": "CON8", **row} for row in history)
    _, lp, risk = head_and_risk(
        train_data,
        test_data,
        contrastive_train,
        contrastive_test,
        alpha,
        horizon,
    )
    predictions["CON8"] = (lp, risk)

    print("  Risk-aware masked AE-8", flush=True)
    risk_train, risk_test, history = risk_aware_finetune(
        masked_model,
        train_z,
        test_z,
        train_data.time_years.to_numpy(float),
        train_data.cvd_death.to_numpy(int),
        train_data.lp_predict.to_numpy(float),
        ai_config,
        seed + 300,
    )
    representations["RAE8"] = (risk_train, risk_test)
    histories.extend({"outer_fold": fold, "method_id": "RAE8", **row} for row in history)
    _, lp, risk = head_and_risk(
        train_data, test_data, risk_train, risk_test, alpha, horizon
    )
    predictions["RAE8"] = (lp, risk)

    payload = {"train_indices": train_indices, "test_indices": test_indices}
    for method_id, (lp, risk) in predictions.items():
        payload[f"{method_id}_lp"] = lp
        payload[f"{method_id}_risk"] = risk
    for method_id, (train_features, test_features) in representations.items():
        payload[f"{method_id}_train_features"] = train_features
        payload[f"{method_id}_features"] = test_features
    np.savez_compressed(cache_path, **payload)
    rows = []
    for method_id, (lp, risk) in predictions.items():
        for local, global_index in enumerate(test_indices):
            rows.append(
                {
                    "SEQN": int(data.iloc[global_index].SEQN),
                    "outer_fold": fold,
                    "method_id": method_id,
                    "linear_predictor": float(lp[local]),
                    "risk_10y": float(risk[local]),
                }
            )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(histories),
        readout_rows(train_data, test_data, representations, fold),
    )


def readout_rows(train_data, test_data, representations, fold):
    rows = []
    for method_id, (train_features, test_features) in representations.items():
        mean = np.mean(train_features, axis=0)
        scale = np.std(train_features, axis=0)
        scale[scale < 1e-8] = 1.0
        train_x = (train_features - mean) / scale
        test_x = (test_features - mean) / scale
        train_design = np.column_stack([np.ones(len(train_x)), train_x])
        test_design = np.column_stack([np.ones(len(test_x)), test_x])
        penalty = np.eye(train_design.shape[1]) * 1e-3
        penalty[0, 0] = 0.0
        for target, column in READOUT_TARGETS.items():
            coefficient = np.linalg.solve(
                train_design.T @ train_design + penalty,
                train_design.T @ train_data[column].to_numpy(float),
            )
            prediction = test_design @ coefficient
            for local, seqn in enumerate(test_data.SEQN.to_numpy(int)):
                rows.append(
                    {
                        "SEQN": seqn,
                        "outer_fold": fold,
                        "method_id": method_id,
                        "target": target,
                        "observed": float(test_data.iloc[local][column]),
                        "predicted": float(prediction[local]),
                    }
                )
    return pd.DataFrame(rows)


def readout_summary(readouts):
    rows = []
    for (method_id, target), group in readouts.groupby(["method_id", "target"]):
        observed = group.observed.to_numpy(float)
        predicted = group.predicted.to_numpy(float)
        denominator = np.sum((observed - observed.mean()) ** 2)
        r2 = 1.0 - np.sum((observed - predicted) ** 2) / denominator
        rows.append(
            {
                "method_id": method_id,
                "method": METHODS[method_id],
                "target": target,
                "cross_validated_R2": float(r2),
            }
        )
    return pd.DataFrame(rows)


def discrimination_summary(data, predictions, replicates, seed):
    lp = predictions.pivot(index="SEQN", columns="method_id", values="linear_predictor").loc[data.SEQN]
    point = {method: c_index(data, lp[method].to_numpy(float)) for method in METHODS}
    samples = {method: [] for method in METHODS}
    delta_base = {method: [] for method in METHODS}
    delta_circadian = {method: [] for method in METHODS}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[indices].reset_index(drop=True)
        values = {
            method: c_index(sample, lp[method].to_numpy(float)[indices])
            for method in METHODS
        }
        for method in METHODS:
            samples[method].append(values[method])
            delta_base[method].append(values[method] - values["B0"])
            delta_circadian[method].append(values[method] - values["B1"])
    rows = []
    for method in METHODS:
        ci_low, ci_high = np.percentile(samples[method], [2.5, 97.5])
        base_low, base_high = np.percentile(delta_base[method], [2.5, 97.5])
        circ_low, circ_high = np.percentile(delta_circadian[method], [2.5, 97.5])
        rows.append(
            {
                "method_id": method,
                "method": METHODS[method],
                "c_index": point[method],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "delta_vs_B0": point[method] - point["B0"],
                "delta_vs_B0_low": base_low,
                "delta_vs_B0_high": base_high,
                "delta_vs_B1": point[method] - point["B1"],
                "delta_vs_B1_low": circ_low,
                "delta_vs_B1_high": circ_high,
            }
        )
    return pd.DataFrame(rows)


def clinical_summary(data, predictions, horizon):
    lp = predictions.pivot(index="SEQN", columns="method_id", values="linear_predictor").loc[data.SEQN]
    risk = predictions.pivot(index="SEQN", columns="method_id", values="risk_10y").loc[data.SEQN]
    values = point_clinical_metrics(
        data,
        {method: lp[method].to_numpy(float) for method in METHODS},
        {method: risk[method].to_numpy(float) for method in METHODS},
        horizon,
    )
    rows = []
    for method, metrics in values.items():
        rows.append({"method_id": method, "method": METHODS[method], **metrics})
    return pd.DataFrame(rows)


def make_figure(results, smoke):
    order = list(METHODS)
    data = results.set_index("method_id").loc[order]
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw={"width_ratios": [1, 1.08]})
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
            xerr=[[row.delta_vs_B1 - row.delta_vs_B1_low], [row.delta_vs_B1_high - row.delta_vs_B1]],
            fmt="o",
            color=COLORS[method],
            capsize=3,
        )
    labels = [METHODS[method].replace("PREDICT LP + ", "+ ") for method in order]
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_xlabel("Out-of-fold C-index (95% CI)")
    axes[0].set_title("A. Same 8-D size, different objectives", loc="left", fontweight="bold")
    axes[1].axvline(0, color="#555555", ls="--", lw=1)
    axes[1].set_xlabel("Paired change vs circadian (95% CI)")
    axes[1].set_title("B. Does AI add beyond rhythm features?", loc="left", fontweight="bold")
    fig.suptitle(
        "Average-day AI representation benchmark",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.01,
        "Exploratory fixed configuration: 8 dimensions, ridge alpha=0.1; all representation learning is outer-fold safe.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    suffix = "_smoke" if smoke else ""
    fig.savefig(FIGURES / f"fig09_ai_representation_benchmark{suffix}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGURES / f"fig09_ai_representation_benchmark{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_readout_figure(readout, smoke):
    methods = list(REPRESENTATION_METHODS)
    targets = list(READOUT_TARGETS)
    matrix = (
        readout.pivot(index="method_id", columns="target", values="cross_validated_R2")
        .loc[methods, targets]
        .to_numpy(float)
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1.0, float(np.nanmax(matrix))))
    for row in range(len(methods)):
        for column in range(len(targets)):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "#222222",
                fontsize=9,
            )
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([METHODS[method].replace("PREDICT LP + ", "") for method in methods])
    ax.set_xlabel("Known activity feature predicted from the 8-D representation")
    ax.set_title("What information does each representation retain?", loc="left", fontsize=13, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Outer-fold readout R2")
    fig.text(
        0.08,
        0.01,
        "Each readout is fitted on outer-training embeddings and evaluated on held-out participants.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    suffix = "_smoke" if smoke else ""
    fig.savefig(FIGURES / f"fig10_ai_representation_readout{suffix}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGURES / f"fig10_ai_representation_readout{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(results, clinical, readout, data, smoke, config):
    indexed = results.set_index("method_id")
    clinical = clinical.set_index("method_id")
    readout = readout.pivot(index="method_id", columns="target", values="cross_validated_R2")
    learned = indexed.loc[["AE8", "MAE8", "CON8", "RAE8"]]
    best = learned.c_index.idxmax()
    row = indexed.loc[best]
    endpoint_label = config.get("endpoint", {}).get("label", "CVD deaths")
    if row.delta_vs_B1_low > 0:
        conclusion = "The paired confidence interval was above zero."
    elif row.delta_vs_B1_high < 0:
        conclusion = "The paired confidence interval was below zero."
    else:
        conclusion = "The paired confidence interval included no improvement."
    text = f"""# Average-day AI representation experiment

This exploratory benchmark used **{len(data):,} participants** and
**{int(data.cvd_death.sum())} {endpoint_label}**. Every method used eight
dimensions, ridge alpha=0.1, the same outer folds, and training-fold-only
representation learning.

## Result

The highest learned-representation point estimate was **{METHODS[best]}** with
C-index **{row.c_index:.4f}**. Its paired change versus circadian features was
**{row.delta_vs_B1:+.4f}** (95% CI {row.delta_vs_B1_low:+.4f} to
{row.delta_vs_B1_high:+.4f}). {conclusion}

PCA-8 achieved C-index **{indexed.loc['P8','c_index']:.4f}** and the standard
reconstruction AE-8 achieved **{indexed.loc['AE8','c_index']:.4f}**. These
comparators show whether any gain comes from nonlinear or objective-specific
learning rather than simply compressing the complete activity profile.

The best learned method had ten-year Brier score
**{clinical.loc[best,'brier_10y']:.5f}** and net benefit at 10% of
**{clinical.loc[best,'net_benefit_10pct']:.5f}**. These are exploratory because
the mortality thresholds are not validated treatment thresholds.

Across methods, volume and M10 were strongly recoverable from the learned
representations. Contrastive-8 recovered MVPA with cross-validated R2
**{readout.loc['CON8','MVPA']:.2f}**, while its L5 and RA readouts were
**{readout.loc['CON8','L5']:.2f}** and **{readout.loc['CON8','RA']:.2f}**.
This indicates that the objectives largely reorganise known activity structure
rather than discovering a clearly independent prognostic phenotype.

## Interpretation rule

This experiment compares AI objectives; it does not select a new primary model
from the outer-test results. A promising objective must subsequently be locked
and validated independently or in a new nested selection analysis.
"""
    target = HERE / ("AI_EXPERIMENT_RESULTS_SMOKE.md" if smoke else "AI_EXPERIMENT_RESULTS.md")
    target.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.smoke)
    data, profiles, folds = analysis_inputs(args.smoke, config)
    prediction_parts, history_parts, readout_parts = [], [], []
    for fold, test_indices in enumerate(folds, start=1):
        prediction, history, readout = run_fold(
            data, profiles, test_indices, fold, config, args.smoke, args.force
        )
        prediction_parts.append(prediction)
        if not history.empty:
            history_parts.append(history)
        readout_parts.append(readout)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["SEQN", "method_id"]
    )
    suffix = "_smoke" if args.smoke else ""
    predictions["method"] = predictions.method_id.map(METHODS)
    predictions.to_parquet(
        PREDICTIONS / f"ai_representation_predictions{suffix}.parquet", index=False
    )
    if history_parts:
        pd.concat(history_parts, ignore_index=True).to_csv(
            TABLES / f"ai_representation_training{suffix}.csv", index=False
        )
    results = discrimination_summary(
        data,
        predictions,
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + 1201,
    )
    clinical = clinical_summary(data, predictions, float(config["horizon_years"]))
    readouts = pd.concat(readout_parts, ignore_index=True)
    readout = readout_summary(readouts)
    results.to_csv(TABLES / f"ai_representation_results{suffix}.csv", index=False)
    clinical.to_csv(TABLES / f"ai_representation_clinical{suffix}.csv", index=False)
    readouts.to_parquet(
        PREDICTIONS / f"ai_representation_readouts{suffix}.parquet", index=False
    )
    readout.to_csv(TABLES / f"ai_representation_readout_r2{suffix}.csv", index=False)
    make_figure(results, args.smoke)
    make_readout_figure(readout, args.smoke)
    write_summary(results, clinical, readout, data, args.smoke, config)
    print(results.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
