"""Repeat the primary and locked-contrastive analyses across outer-fold splits.

This sensitivity analysis does not replace the canonical five-fold result. It
measures how much paired C-index changes when the participant partition is
changed while every modelling rule remains fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_representation_models import train_contrastive
from models import (
    CACHE,
    DATA,
    FIGURES,
    HERE,
    PREDICTIONS,
    TABLES,
    c_index,
    fit_head,
    fit_selected_model,
    load_config,
    stratified_folds,
)


FULL_SPLIT_SEEDS = (2026, 2127, 2228, 2329, 2430)
CIRCADIAN_COLUMNS = ("M10", "L5", "RA")
COMPARISONS = {
    "M6_vs_M3": (
        "m6_lp",
        "m3_lp",
        "Nested AE beyond activity-rhythm features",
    ),
    "CCON8_vs_B1": (
        "ccon8_lp",
        "b1_lp",
        "Locked contrastive-8 beyond activity-rhythm features",
    ),
}


def load_inputs(smoke: bool, config: dict) -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_parquet(DATA / "cohort.parquet")
    with np.load(DATA / "average_day_profiles.npz") as stored:
        seqn = stored["seqn"].astype(int)
        profiles = stored["profile"].astype(np.float32)
    if not np.array_equal(seqn, data.SEQN.to_numpy(int)):
        raise ValueError("Cohort and profile participant order disagree")
    if smoke:
        rng = np.random.default_rng(99)
        events = np.flatnonzero(data.cvd_death.to_numpy(int) == 1)
        nonevents = np.flatnonzero(data.cvd_death.to_numpy(int) == 0)
        selected = np.sort(
            np.concatenate(
                [events, rng.choice(nonevents, int(config["smoke"]["non_events"]), replace=False)]
            )
        )
        data = data.iloc[selected].reset_index(drop=True)
        profiles = profiles[selected]
    return data, profiles


def ai_config(config: dict, smoke: bool) -> dict:
    values = dict(config["ai_experiment"])
    if smoke:
        values["epochs"] = int(config["smoke"]["ai_epochs"])
    return values


def cache_path(split_seed: int, fold: int, smoke: bool) -> Path:
    suffix = "smoke" if smoke else "full"
    return CACHE / f"repeated_split_{suffix}_seed_{split_seed}_fold_{fold}.npz"


def canonical_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Reuse the already-audited canonical split rather than refitting it."""
    canonical = pd.read_parquet(PREDICTIONS / "oof_predictions.parquet")
    ai = pd.read_parquet(PREDICTIONS / "ai_incremental_predictions.parquet")
    index_columns = ["SEQN", "outer_fold"]
    model_columns = [
        *index_columns,
        "linear_predictor",
        "selected_dimension",
        "selected_alpha",
    ]
    m3 = canonical[canonical.model_id == "M3"][model_columns].rename(
        columns={
            "linear_predictor": "m3_lp",
            "selected_dimension": "m3_dimension",
            "selected_alpha": "m3_alpha",
        }
    )
    m6 = canonical[canonical.model_id == "M6"][model_columns].rename(
        columns={
            "linear_predictor": "m6_lp",
            "selected_dimension": "m6_dimension",
            "selected_alpha": "m6_alpha",
        }
    )
    b1 = ai[ai.method_id == "B1"][index_columns + ["linear_predictor"]].rename(
        columns={"linear_predictor": "b1_lp"}
    )
    ccon8 = ai[ai.method_id == "CCON8"][index_columns + ["linear_predictor"]].rename(
        columns={"linear_predictor": "ccon8_lp"}
    )
    result = m3.merge(m6, on=index_columns, validate="one_to_one")
    result = result.merge(b1, on=index_columns, validate="one_to_one")
    result = result.merge(ccon8, on=index_columns, validate="one_to_one")
    metadata = data.set_index("SEQN")
    result["split_seed"] = FULL_SPLIT_SEEDS[0]
    result["cycle"] = result.SEQN.map(metadata.cycle)
    result["time_years"] = result.SEQN.map(metadata.time_years)
    result["cvd_death"] = result.SEQN.map(metadata.cvd_death)
    return result


def run_fold(
    data: pd.DataFrame,
    profiles: np.ndarray,
    test_indices: np.ndarray,
    split_seed: int,
    fold: int,
    config: dict,
    smoke: bool,
    force: bool,
) -> pd.DataFrame:
    target = cache_path(split_seed, fold, smoke)
    if target.exists() and not force:
        with np.load(target) as stored:
            test_indices = stored["test_indices"].astype(int)
            payload = {key: stored[key] for key in stored.files if key != "test_indices"}
        return fold_rows(data, test_indices, split_seed, fold, payload)

    all_indices = np.arange(len(data))
    train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
    train_data = data.iloc[train_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    train_profiles = profiles[train_indices]
    test_profiles = profiles[test_indices]
    print(
        f"Split {split_seed}, fold {fold}: "
        f"train={len(train_data):,}/{int(train_data.cvd_death.sum())} events; "
        f"test={len(test_data):,}/{int(test_data.cvd_death.sum())} events",
        flush=True,
    )

    print("  Nested M3", flush=True)
    m3 = fit_selected_model(
        train_data,
        test_data,
        train_profiles,
        test_profiles,
        "M3",
        config,
        split_seed + fold * 10000 + 3000,
    )
    print("  Nested M6", flush=True)
    m6 = fit_selected_model(
        train_data,
        test_data,
        train_profiles,
        test_profiles,
        "M6",
        config,
        split_seed + fold * 10000 + 6000,
    )

    fixed_alpha = float(config["ai_experiment"]["ridge_alpha"])
    train_circadian = train_data[list(CIRCADIAN_COLUMNS)].to_numpy(float)
    test_circadian = test_data[list(CIRCADIAN_COLUMNS)].to_numpy(float)
    _, b1_lp, _ = fit_head(
        train_data,
        test_data,
        train_circadian,
        test_circadian,
        fixed_alpha,
    )
    print("  Locked contrastive-8", flush=True)
    contrastive_train, contrastive_test, _ = train_contrastive(
        train_profiles,
        test_profiles,
        ai_config(config, smoke),
        split_seed + fold * 10000 + 1400,
    )
    _, ccon8_lp, _ = fit_head(
        train_data,
        test_data,
        np.column_stack([train_circadian, contrastive_train]),
        np.column_stack([test_circadian, contrastive_test]),
        fixed_alpha,
    )

    payload = {
        "m3_lp": m3["test_lp"],
        "m6_lp": m6["test_lp"],
        "b1_lp": b1_lp,
        "ccon8_lp": ccon8_lp,
        "m3_dimension": np.asarray([m3["selected"].dimension]),
        "m6_dimension": np.asarray([m6["selected"].dimension]),
        "m3_alpha": np.asarray([m3["selected"].alpha]),
        "m6_alpha": np.asarray([m6["selected"].alpha]),
    }
    np.savez_compressed(target, test_indices=test_indices, **payload)
    return fold_rows(data, test_indices, split_seed, fold, payload)


def fold_rows(
    data: pd.DataFrame,
    test_indices: np.ndarray,
    split_seed: int,
    fold: int,
    payload: dict,
) -> pd.DataFrame:
    rows = []
    scalars = {
        key: float(np.asarray(payload[key]).reshape(-1)[0])
        for key in ("m3_dimension", "m6_dimension", "m3_alpha", "m6_alpha")
    }
    for local, global_index in enumerate(test_indices):
        participant = data.iloc[global_index]
        rows.append(
            {
                "SEQN": int(participant.SEQN),
                "cycle": str(participant.cycle),
                "time_years": float(participant.time_years),
                "cvd_death": int(participant.cvd_death),
                "split_seed": int(split_seed),
                "outer_fold": int(fold),
                "m3_lp": float(payload["m3_lp"][local]),
                "m6_lp": float(payload["m6_lp"][local]),
                "b1_lp": float(payload["b1_lp"][local]),
                "ccon8_lp": float(payload["ccon8_lp"][local]),
                **scalars,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(
    data: pd.DataFrame,
    candidate: np.ndarray,
    comparator: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    point_candidate = c_index(data, candidate)
    point_comparator = c_index(data, comparator)
    deltas = []
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[indices].reset_index(drop=True)
        deltas.append(
            c_index(sample, candidate[indices]) - c_index(sample, comparator[indices])
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "candidate_c_index": point_candidate,
        "comparator_c_index": point_comparator,
        "delta_c": point_candidate - point_comparator,
        "delta_low": float(low),
        "delta_high": float(high),
        "bootstrap_probability_positive": float(np.mean(np.asarray(deltas) > 0)),
    }


def summarize(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for split_seed, group in predictions.groupby("split_seed", sort=True):
        ordered = group.set_index("SEQN").loc[data.SEQN]
        for comparison, (candidate_column, comparator_column, label) in COMPARISONS.items():
            rows.append(
                {
                    "split_seed": int(split_seed),
                    "comparison": comparison,
                    "comparison_label": label,
                    **paired_bootstrap(
                        data,
                        ordered[candidate_column].to_numpy(float),
                        ordered[comparator_column].to_numpy(float),
                        int(config["bootstrap_replicates"]),
                        int(split_seed) + (2000 if comparison == "M6_vs_M3" else 3000),
                    ),
                }
            )
    results = pd.DataFrame(rows)
    summaries = []
    for comparison, group in results.groupby("comparison", sort=False):
        values = group.delta_c.to_numpy(float)
        summaries.append(
            {
                "comparison": comparison,
                "comparison_label": group.comparison_label.iloc[0],
                "split_count": len(group),
                "mean_delta_c": float(values.mean()),
                "sd_delta_c": float(values.std(ddof=1)),
                "minimum_delta_c": float(values.min()),
                "maximum_delta_c": float(values.max()),
                "positive_splits": int((values > 0).sum()),
            }
        )
    return results, pd.DataFrame(summaries)


def make_figure(results: pd.DataFrame, smoke: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    colors = {"M6_vs_M3": "#009E73", "CCON8_vs_B1": "#D55E00"}
    for axis, comparison in zip(axes, COMPARISONS):
        group = results[results.comparison == comparison].sort_values("split_seed")
        positions = np.arange(1, len(group) + 1)
        axis.axhline(0, color="#555555", ls="--", lw=1)
        axis.errorbar(
            positions,
            group.delta_c,
            yerr=[group.delta_c - group.delta_low, group.delta_high - group.delta_c],
            fmt="o",
            color=colors[comparison],
            capsize=3,
            lw=1.2,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([f"Split {index}" for index in positions])
        axis.set_xlabel("Outer-fold partition")
        axis.grid(axis="y", alpha=0.2)
        axis.set_title(COMPARISONS[comparison][2], loc="left", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Paired C-index change")
    fig.suptitle(
        "Sensitivity to participant-fold assignment",
        x=0.07,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.01,
        "Dots are complete out-of-fold estimates; bars are participant-bootstrap 95% intervals conditional on each fitted split.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    suffix = "_smoke" if smoke else ""
    fig.savefig(FIGURES / f"fig17_repeated_split_stability{suffix}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGURES / f"fig17_repeated_split_stability{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(results: pd.DataFrame, summary: pd.DataFrame, smoke: bool) -> None:
    lines = [
        "# Repeated outer-split stability",
        "",
        "This sensitivity analysis changes only the participant-fold partition. Model",
        "selection rules, representation definitions, ridge grid and evaluation remain fixed.",
        "It measures split and training instability; it is not external validation.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.extend(
            [
                f"## {row.comparison_label}",
                "",
                f"Across {row.split_count} partitions, mean paired delta C was "
                f"**{row.mean_delta_c:+.5f}** (SD {row.sd_delta_c:.5f}; range "
                f"{row.minimum_delta_c:+.5f} to {row.maximum_delta_c:+.5f}). "
                f"The point estimate was positive in **{row.positive_splits}/{row.split_count}** partitions.",
                "",
            ]
        )
    lines.extend(
        [
            "## Split-level results",
            "",
            "| Split seed | Comparison | Candidate C | Comparator C | Delta C | 95% interval |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.split_seed} | {row.comparison} | {row.candidate_c_index:.5f} | "
            f"{row.comparator_c_index:.5f} | {row.delta_c:+.5f} | "
            f"{row.delta_low:+.5f} to {row.delta_high:+.5f} |"
        )
    lines.append("")
    target = HERE / ("REPEATED_SPLIT_STABILITY_SMOKE.md" if smoke else "REPEATED_SPLIT_STABILITY.md")
    target.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.smoke)
    data, profiles = load_inputs(args.smoke, config)
    split_seeds = FULL_SPLIT_SEEDS[:2] if args.smoke else FULL_SPLIT_SEEDS
    if args.split_seed is not None:
        if args.split_seed not in split_seeds:
            raise ValueError(f"--split-seed must be one of {split_seeds}")
        split_seeds = (args.split_seed,)

    suffix = "_smoke" if args.smoke else ""
    existing_path = PREDICTIONS / f"repeated_split_predictions{suffix}.parquet"
    existing = pd.read_parquet(existing_path) if existing_path.exists() and not args.force else pd.DataFrame()
    all_rows = [] if existing.empty else [existing]
    completed = set()
    if not existing.empty:
        completed = set(zip(existing.split_seed.astype(int), existing.outer_fold.astype(int)))

    if not args.smoke and FULL_SPLIT_SEEDS[0] in split_seeds and not args.force:
        if not any(seed == FULL_SPLIT_SEEDS[0] for seed, _ in completed):
            canonical = canonical_rows(data)
            all_rows.append(canonical)
            completed.update(zip(canonical.split_seed.astype(int), canonical.outer_fold.astype(int)))

    for split_seed in split_seeds:
        folds = stratified_folds(data, int(config["outer_folds"]), int(split_seed))
        fold_numbers = [args.fold] if args.fold is not None else list(range(1, len(folds) + 1))
        for fold in fold_numbers:
            if fold is None or not 1 <= fold <= len(folds):
                raise ValueError(f"--fold must be between 1 and {len(folds)}")
            if (int(split_seed), int(fold)) in completed and not args.force:
                continue
            all_rows.append(
                run_fold(
                    data,
                    profiles,
                    folds[fold - 1],
                    int(split_seed),
                    int(fold),
                    config,
                    args.smoke,
                    args.force,
                )
            )

    predictions = (
        pd.concat(all_rows, ignore_index=True)
        .drop_duplicates(["SEQN", "split_seed"], keep="last")
        .sort_values(["split_seed", "SEQN"])
        .reset_index(drop=True)
    )
    predictions.to_parquet(existing_path, index=False)
    expected = len(split_seeds) * len(data)
    available = predictions[predictions.split_seed.isin(split_seeds)]
    if len(available) != expected:
        print(
            f"Saved partial repeated-split predictions ({len(available):,}/{expected:,} rows); "
            "run remaining folds before summarising.",
            flush=True,
        )
        return

    results, summary = summarize(data, available, config)
    results.to_csv(TABLES / f"repeated_split_stability{suffix}.csv", index=False)
    summary.to_csv(TABLES / f"repeated_split_summary{suffix}.csv", index=False)
    make_figure(results, args.smoke)
    write_summary(results, summary, args.smoke)
    print("\nRepeated-split summary")
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
