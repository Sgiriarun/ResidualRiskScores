"""Validate the locked circadian + contrastive-8 candidate across seeds and cycles."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_representation_models import train_contrastive
from models import CACHE, DATA, FIGURES, HERE, PREDICTIONS, TABLES, c_index, fit_head, load_config


SEED_COUNT = 5
CIRCADIAN_COLUMNS = ["M10", "L5", "RA"]
COLORS = {"B0": "#7A8594", "B1": "#E69F00", "CCON8": "#D55E00"}
LABELS = {
    "B0": "Transported PREDICT LP",
    "B1": "PREDICT LP + circadian",
    "CCON8": "PREDICT LP + circadian + contrastive-8",
}


def load_analysis(smoke: bool):
    data = pd.read_parquet(DATA / "cohort.parquet")
    with np.load(DATA / "average_day_profiles.npz") as stored:
        profiles = stored["profile"].astype(np.float32)
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
        profiles = profiles[selected]
        predictions = pd.read_parquet(PREDICTIONS / "ai_representation_predictions_smoke.parquet")
    else:
        predictions = pd.read_parquet(PREDICTIONS / "ai_representation_predictions.parquet")
    fold_by_seqn = (
        predictions[["SEQN", "outer_fold"]]
        .drop_duplicates("SEQN")
        .set_index("SEQN")
        .loc[data.SEQN, "outer_fold"]
        .to_numpy(int)
    )
    return data, profiles, fold_by_seqn


def ai_config(config: dict, smoke: bool):
    values = dict(config["ai_experiment"])
    if smoke:
        values["epochs"] = int(config["smoke"]["ai_epochs"])
    return values


def training_seed(config_seed: int, seed_id: int, fold: int) -> int:
    return config_seed + fold * 10000 + 1400 + (seed_id - 1) * 100000


def stability_embedding(
    train_profiles,
    test_profiles,
    train_indices,
    test_indices,
    fold,
    seed_id,
    config,
    smoke,
    force,
):
    suffix = "smoke" if smoke else "full"
    cache_path = CACHE / f"contrastive_stability_{suffix}_seed_{seed_id}_fold_{fold}.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path) as stored:
            return stored["train_embedding"], stored["test_embedding"]

    # Seed 1 exactly reproduces the representation used in the objective benchmark.
    original_path = CACHE / f"ai_representation_{suffix}_fold_{fold}.npz"
    if seed_id == 1 and original_path.exists() and not force:
        with np.load(original_path) as stored:
            train_embedding = stored["CON8_train_features"]
            test_embedding = stored["CON8_features"]
    else:
        seed = training_seed(int(config["seed"]), seed_id, fold)
        train_embedding, test_embedding, _ = train_contrastive(
            train_profiles, test_profiles, ai_config(config, smoke), seed
        )
    np.savez_compressed(
        cache_path,
        train_indices=train_indices,
        test_indices=test_indices,
        train_embedding=train_embedding,
        test_embedding=test_embedding,
    )
    return train_embedding, test_embedding


def run_stability(data, profiles, fold_by_seqn, config, smoke, force):
    seed_count = 2 if smoke else SEED_COUNT
    alpha = float(config["ai_experiment"]["ridge_alpha"])
    rows = []
    all_indices = np.arange(len(data))
    for seed_id in range(1, seed_count + 1):
        for fold in sorted(np.unique(fold_by_seqn)):
            test_indices = np.flatnonzero(fold_by_seqn == fold)
            train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
            train_data = data.iloc[train_indices].reset_index(drop=True)
            test_data = data.iloc[test_indices].reset_index(drop=True)
            print(f"Stability seed {seed_id}/{seed_count}, fold {fold}", flush=True)
            train_embedding, test_embedding = stability_embedding(
                profiles[train_indices],
                profiles[test_indices],
                train_indices,
                test_indices,
                int(fold),
                seed_id,
                config,
                smoke,
                force,
            )
            train_features = np.column_stack(
                [train_data[CIRCADIAN_COLUMNS].to_numpy(float), train_embedding]
            )
            test_features = np.column_stack(
                [test_data[CIRCADIAN_COLUMNS].to_numpy(float), test_embedding]
            )
            _, test_lp, _ = fit_head(
                train_data, test_data, train_features, test_features, alpha
            )
            for local, global_index in enumerate(test_indices):
                rows.append(
                    {
                        "SEQN": int(data.iloc[global_index].SEQN),
                        "outer_fold": int(fold),
                        "seed_id": seed_id,
                        "training_seed": training_seed(
                            int(config["seed"]), seed_id, int(fold)
                        ),
                        "linear_predictor": float(test_lp[local]),
                    }
                )
    return pd.DataFrame(rows)


def paired_bootstrap(data, candidate, baseline, replicates, seed):
    point_candidate = c_index(data, candidate)
    point_baseline = c_index(data, baseline)
    candidate_samples, deltas = [], []
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[indices].reset_index(drop=True)
        candidate_c = c_index(sample, candidate[indices])
        baseline_c = c_index(sample, baseline[indices])
        candidate_samples.append(candidate_c)
        deltas.append(candidate_c - baseline_c)
    c_low, c_high = np.percentile(candidate_samples, [2.5, 97.5])
    d_low, d_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "c_index": point_candidate,
        "c_index_low": c_low,
        "c_index_high": c_high,
        "baseline_c_index": point_baseline,
        "delta_vs_circadian": point_candidate - point_baseline,
        "delta_low": d_low,
        "delta_high": d_high,
        "bootstrap_probability_positive": float(np.mean(np.asarray(deltas) > 0)),
    }


def stability_summary(data, predictions, config, smoke):
    original = pd.read_parquet(
        PREDICTIONS
        / ("ai_representation_predictions_smoke.parquet" if smoke else "ai_representation_predictions.parquet")
    )
    baseline = (
        original[original.method_id == "B1"].set_index("SEQN").loc[data.SEQN, "linear_predictor"].to_numpy(float)
    )
    rows = []
    for seed_id, group in predictions.groupby("seed_id"):
        candidate = group.set_index("SEQN").loc[data.SEQN, "linear_predictor"].to_numpy(float)
        rows.append(
            {
                "seed_id": int(seed_id),
                **paired_bootstrap(
                    data,
                    candidate,
                    baseline,
                    int(config["bootstrap_replicates"]),
                    int(config["seed"]) + 1400 + int(seed_id),
                ),
            }
        )
    return pd.DataFrame(rows)


def transport_embeddings(train_profiles, test_profiles, direction, seed_id, config, smoke, force):
    suffix = "smoke" if smoke else "full"
    cache_path = CACHE / f"contrastive_transport_{suffix}_{direction}_seed_{seed_id}.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path) as stored:
            return stored["train_embedding"], stored["test_embedding"]
    direction_code = sum(ord(value) for value in direction)
    seed = int(config["seed"]) + 500000 + direction_code * 100 + seed_id * 10000
    train_embedding, test_embedding, _ = train_contrastive(
        train_profiles, test_profiles, ai_config(config, smoke), seed
    )
    np.savez_compressed(
        cache_path,
        train_embedding=train_embedding,
        test_embedding=test_embedding,
        training_seed=np.asarray([seed]),
    )
    return train_embedding, test_embedding


def run_transport(data, profiles, config, smoke, force):
    seed_count = 2 if smoke else SEED_COUNT
    alpha = float(config["ai_experiment"]["ridge_alpha"])
    prediction_rows, summary_rows = [], []
    for train_cycle, test_cycle in (("C", "D"), ("D", "C")):
        train_indices = np.flatnonzero(data.cycle.to_numpy(str) == train_cycle)
        test_indices = np.flatnonzero(data.cycle.to_numpy(str) == test_cycle)
        train_data = data.iloc[train_indices].reset_index(drop=True)
        test_data = data.iloc[test_indices].reset_index(drop=True)
        train_circadian = train_data[CIRCADIAN_COLUMNS].to_numpy(float)
        test_circadian = test_data[CIRCADIAN_COLUMNS].to_numpy(float)
        _, baseline_lp, _ = fit_head(
            train_data, test_data, train_circadian, test_circadian, alpha
        )
        predict_lp = test_data.lp_predict.to_numpy(float)
        direction = f"{train_cycle}_to_{test_cycle}"
        for seed_id in range(1, seed_count + 1):
            print(f"Transport {train_cycle}->{test_cycle}, seed {seed_id}/{seed_count}", flush=True)
            train_embedding, test_embedding = transport_embeddings(
                profiles[train_indices],
                profiles[test_indices],
                direction,
                seed_id,
                config,
                smoke,
                force,
            )
            train_features = np.column_stack([train_circadian, train_embedding])
            test_features = np.column_stack([test_circadian, test_embedding])
            _, candidate_lp, _ = fit_head(
                train_data, test_data, train_features, test_features, alpha
            )
            summary_rows.append(
                {
                    "train_cycle": train_cycle,
                    "test_cycle": test_cycle,
                    "test_participants": len(test_data),
                    "test_events": int(test_data.cvd_death.sum()),
                    "seed_id": seed_id,
                    "predict_c_index": c_index(test_data, predict_lp),
                    **paired_bootstrap(
                        test_data,
                        candidate_lp,
                        baseline_lp,
                        int(config["bootstrap_replicates"]),
                        int(config["seed"]) + 1500 + ord(train_cycle) + seed_id,
                    ),
                }
            )
            for local, global_index in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "SEQN": int(data.iloc[global_index].SEQN),
                        "train_cycle": train_cycle,
                        "test_cycle": test_cycle,
                        "seed_id": seed_id,
                        "predict_lp": float(predict_lp[local]),
                        "circadian_lp": float(baseline_lp[local]),
                        "circadian_contrastive_lp": float(candidate_lp[local]),
                    }
                )
    return pd.DataFrame(prediction_rows), pd.DataFrame(summary_rows)


def make_figure(stability, transport, smoke):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), gridspec_kw={"width_ratios": [1, 1.15]})
    axes[0].axhline(0, color="#555555", ls="--", lw=1)
    axes[0].errorbar(
        stability.seed_id,
        stability.delta_vs_circadian,
        yerr=[
            stability.delta_vs_circadian - stability.delta_low,
            stability.delta_high - stability.delta_vs_circadian,
        ],
        fmt="o",
        color=COLORS["CCON8"],
        capsize=3,
    )
    axes[0].set_xticks(stability.seed_id)
    axes[0].set_xlabel("Independent training seed")
    axes[0].set_ylabel("Paired C-index change vs circadian")
    axes[0].set_title("A. Training-seed stability", loc="left", fontweight="bold")

    directions = [("C", "D"), ("D", "C")]
    y_positions = [1, 0]
    all_transport_values = transport.delta_vs_circadian.to_numpy(float)
    value_span = max(all_transport_values.max() - min(0, all_transport_values.min()), 0.001)
    for y, (train_cycle, test_cycle) in zip(y_positions, directions):
        group = transport[
            (transport.train_cycle == train_cycle) & (transport.test_cycle == test_cycle)
        ].sort_values("seed_id")
        values = group.delta_vs_circadian.to_numpy(float)
        jitter = np.linspace(-0.11, 0.11, len(group))
        axes[1].hlines(
            y,
            values.min(),
            values.max(),
            color="#AAB2BC",
            lw=3,
            zorder=1,
        )
        axes[1].scatter(
            values,
            np.full(len(group), y) + jitter,
            color=COLORS["CCON8"],
            s=38,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        mean = values.mean()
        axes[1].scatter(
            mean,
            y,
            marker="D",
            s=70,
            color="#222222",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axes[1].text(
            values.max() + 0.08 * value_span,
            y,
            f"mean {mean:+.4f}  |  {int((values > 0).sum())}/{len(values)} positive",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#303842",
        )
    axes[1].axvline(0, color="#555555", ls="--", lw=1)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(["Train C -> test D", "Train D -> test C"])
    axes[1].set_xlabel("Paired C-index change vs circadian")
    axes[1].set_title(
        "B. Cycle transfer: positive point estimates",
        loc="left",
        fontweight="bold",
    )
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].text(
        0.01,
        -0.25,
        "Dots = training seeds   Diamond = mean   Line = seed range",
        transform=axes[1].transAxes,
        fontsize=8,
        color="#555555",
    )
    axes[0].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Locked circadian + contrastive-8 validation",
        x=0.07,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.01,
        "Five seeds; fixed 8-D representation and ridge alpha=0.1. No seed was selected using test performance.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    suffix = "_smoke" if smoke else ""
    fig.savefig(FIGURES / f"fig12_contrastive_stability_transport{suffix}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGURES / f"fig12_contrastive_stability_transport{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(stability, transport, data, smoke):
    direction_lines = []
    for (train_cycle, test_cycle), group in transport.groupby(["train_cycle", "test_cycle"]):
        direction_lines.append(
            f"- Train {train_cycle}, test {test_cycle}: mean delta "
            f"{group.delta_vs_circadian.mean():+.5f}, range "
            f"{group.delta_vs_circadian.min():+.5f} to {group.delta_vs_circadian.max():+.5f}."
        )
    text = f"""# Locked contrastive validation

The fixed candidate was PREDICT LP + M10/L5/RA + contrastive-8 with ridge
alpha=0.1. No model or seed was selected using the results below.

## Random-seed stability

Across {len(stability)} independent training seeds, mean out-of-fold C-index was
**{stability.c_index.mean():.5f}** (SD {stability.c_index.std(ddof=1):.5f}). Mean
paired change versus circadian alone was
**{stability.delta_vs_circadian.mean():+.5f}**, ranging from
{stability.delta_vs_circadian.min():+.5f} to
{stability.delta_vs_circadian.max():+.5f}. A positive point estimate occurred
for **{int((stability.delta_vs_circadian > 0).sum())}/{len(stability)}** seeds.

## Cycle-held-out transport

{chr(10).join(direction_lines)}

Cycle transport is a difficult test because representation learning, scaling
and the Cox correction are fitted using only the source cycle. Consistent
positive changes in both directions would support transportability; conflicting
directions indicate cycle-specific signal.

## Interpretation

This experiment tests computational stability and temporal-cycle transport. It
does not provide independent external validation because both cycles belong to
NHANES and use the same data-collection programme.
"""
    target = HERE / ("CONTRASTIVE_VALIDATION_SMOKE.md" if smoke else "CONTRASTIVE_VALIDATION.md")
    target.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the figure and summary from existing result tables.",
    )
    args = parser.parse_args()
    config = load_config(args.smoke)
    data, profiles, fold_by_seqn = load_analysis(args.smoke)
    suffix = "_smoke" if args.smoke else ""

    if args.plot_only:
        stability = pd.read_csv(TABLES / f"contrastive_seed_stability{suffix}.csv")
        transport = pd.read_csv(TABLES / f"contrastive_cycle_transport{suffix}.csv")
        make_figure(stability, transport, args.smoke)
        write_summary(stability, transport, data, args.smoke)
        print(f"Figure regenerated from existing tables in {FIGURES}")
        return

    stability_predictions = run_stability(
        data, profiles, fold_by_seqn, config, args.smoke, args.force
    )
    stability_predictions.to_parquet(
        PREDICTIONS / f"contrastive_seed_predictions{suffix}.parquet", index=False
    )
    stability = stability_summary(data, stability_predictions, config, args.smoke)
    stability.to_csv(TABLES / f"contrastive_seed_stability{suffix}.csv", index=False)

    transport_predictions, transport = run_transport(
        data, profiles, config, args.smoke, args.force
    )
    transport_predictions.to_parquet(
        PREDICTIONS / f"contrastive_cycle_transport_predictions{suffix}.parquet", index=False
    )
    transport.to_csv(TABLES / f"contrastive_cycle_transport{suffix}.csv", index=False)
    make_figure(stability, transport, args.smoke)
    write_summary(stability, transport, data, args.smoke)
    print("\nSeed stability")
    print(stability.round(5).to_string(index=False))
    print("\nCycle transport")
    print(transport.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
