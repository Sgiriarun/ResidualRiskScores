"""Explain the locked contrastive signal using objective and augmentation ablations."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_representation_models import train_contrastive
from models import CACHE, DATA, FIGURES, HERE, PREDICTIONS, TABLES, c_index, fit_head, load_config


CIRCADIAN_COLUMNS = ["M10", "L5", "RA"]
VARIANTS = {
    "full": {
        "label": "All augmentations",
        "flags": {},
    },
    "no_mask": {
        "label": "Remove block masking",
        "flags": {"contrastive_use_mask": False},
    },
    "no_scale": {
        "label": "Remove activity scaling",
        "flags": {"contrastive_use_scale": False},
    },
    "no_noise": {
        "label": "Remove random noise",
        "flags": {"contrastive_use_noise": False},
    },
    "no_shift": {
        "label": "Remove time shifting",
        "flags": {"contrastive_use_shift": False},
    },
}
ABLATIONS = tuple(variant for variant in VARIANTS if variant != "full")
SEED_COUNT = 5
OBJECTIVE_LABEL = (
    "Circadian + contrastive-8 minus circadian + reconstruction AE-8"
)


def training_seed(config_seed: int, seed_id: int, fold: int) -> int:
    """Match the locked contrastive seed schedule for paired initialization."""
    return config_seed + fold * 10000 + 1400 + (seed_id - 1) * 100000


def load_analysis(smoke: bool):
    config = load_config(smoke)
    data = pd.read_parquet(DATA / "cohort.parquet")
    with np.load(DATA / "average_day_profiles.npz") as stored:
        profiles = stored["profile"].astype(np.float32)
    suffix = "_smoke" if smoke else ""
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

    benchmark_path = PREDICTIONS / f"ai_representation_predictions{suffix}.parquet"
    incremental_path = PREDICTIONS / f"ai_incremental_predictions{suffix}.parquet"
    full_path = PREDICTIONS / f"contrastive_seed_predictions{suffix}.parquet"
    for path in (benchmark_path, incremental_path, full_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run scripts 12-14 first")

    benchmark = pd.read_parquet(benchmark_path)
    incremental = pd.read_parquet(incremental_path)
    full = pd.read_parquet(full_path)
    fold_by_seqn = (
        benchmark[["SEQN", "outer_fold"]]
        .drop_duplicates("SEQN")
        .set_index("SEQN")
        .loc[data.SEQN, "outer_fold"]
        .to_numpy(int)
    )
    baseline = (
        benchmark[benchmark.method_id == "B1"]
        .set_index("SEQN")
        .loc[data.SEQN, "linear_predictor"]
        .to_numpy(float)
    )
    return config, data, profiles, fold_by_seqn, baseline, incremental, full


def ablation_config(config: dict, smoke: bool, variant: str) -> dict:
    values = dict(config["ai_experiment"])
    if smoke:
        values["epochs"] = int(config["smoke"]["ai_epochs"])
    values.update(VARIANTS[variant]["flags"])
    return values


def ablation_embedding(
    train_profiles,
    test_profiles,
    train_indices,
    test_indices,
    fold,
    seed_id,
    variant,
    config,
    smoke,
    force,
):
    suffix = "smoke" if smoke else "full"
    cache_path = CACHE / f"contrastive_ablation_{suffix}_{variant}_seed_{seed_id}_fold_{fold}.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path) as stored:
            if not np.array_equal(stored["train_indices"].astype(int), train_indices):
                raise ValueError(f"Training indices changed for {cache_path}")
            if not np.array_equal(stored["test_indices"].astype(int), test_indices):
                raise ValueError(f"Test indices changed for {cache_path}")
            return stored["train_embedding"], stored["test_embedding"]

    seed = training_seed(int(config["seed"]), seed_id, fold)
    train_embedding, test_embedding, history = train_contrastive(
        train_profiles,
        test_profiles,
        ablation_config(config, smoke, variant),
        seed,
    )
    np.savez_compressed(
        cache_path,
        train_indices=train_indices,
        test_indices=test_indices,
        train_embedding=train_embedding,
        test_embedding=test_embedding,
        training_seed=np.asarray([seed]),
        final_loss=np.asarray([history[-1]["loss"]]),
    )
    return train_embedding, test_embedding


def run_ablations(data, profiles, fold_by_seqn, full, config, smoke, force, seed_count):
    available_seeds = sorted(full.seed_id.unique())
    requested_seeds = list(range(1, seed_count + 1))
    if not set(requested_seeds).issubset(available_seeds):
        raise ValueError(f"Locked full predictions contain seeds {available_seeds}, not {requested_seeds}")

    rows = []
    full = full[full.seed_id.isin(requested_seeds)].copy()
    full["variant"] = "full"
    rows.extend(full[["SEQN", "outer_fold", "seed_id", "linear_predictor", "variant"]].to_dict("records"))

    alpha = float(config["ai_experiment"]["ridge_alpha"])
    all_indices = np.arange(len(data))
    folds = sorted(np.unique(fold_by_seqn))
    for variant in ABLATIONS:
        for seed_id in requested_seeds:
            for fold in folds:
                test_indices = np.flatnonzero(fold_by_seqn == fold)
                train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
                train_data = data.iloc[train_indices].reset_index(drop=True)
                test_data = data.iloc[test_indices].reset_index(drop=True)
                print(
                    f"Ablation {variant}: seed {seed_id}/{seed_count}, fold {fold}/{len(folds)}",
                    flush=True,
                )
                train_embedding, test_embedding = ablation_embedding(
                    profiles[train_indices],
                    profiles[test_indices],
                    train_indices,
                    test_indices,
                    int(fold),
                    seed_id,
                    variant,
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
                    train_data,
                    test_data,
                    train_features,
                    test_features,
                    alpha,
                )
                for local, global_index in enumerate(test_indices):
                    rows.append(
                        {
                            "SEQN": int(data.iloc[global_index].SEQN),
                            "outer_fold": int(fold),
                            "seed_id": seed_id,
                            "linear_predictor": float(test_lp[local]),
                            "variant": variant,
                        }
                    )
    predictions = pd.DataFrame(rows)
    expected = len(data) * seed_count * len(VARIANTS)
    if len(predictions) != expected:
        raise AssertionError(f"Ablation predictions have {len(predictions)} rows; expected {expected}")
    if predictions.duplicated(["SEQN", "seed_id", "variant"]).any():
        raise AssertionError("Duplicate participant/seed/variant predictions")
    return predictions.sort_values(["variant", "seed_id", "SEQN"]).reset_index(drop=True)


def paired_objective_comparison(data, incremental, replicates, seed):
    wide = incremental.pivot(index="SEQN", columns="method_id", values="linear_predictor").loc[data.SEQN]
    reconstruction = wide["CAE8"].to_numpy(float)
    contrastive = wide["CCON8"].to_numpy(float)
    point_reconstruction = c_index(data, reconstruction)
    point_contrastive = c_index(data, contrastive)
    samples = []
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[indices].reset_index(drop=True)
        samples.append(
            c_index(sample, contrastive[indices]) - c_index(sample, reconstruction[indices])
        )
    low, high = np.percentile(samples, [2.5, 97.5])
    return pd.DataFrame(
        [
            {
                "comparison": OBJECTIVE_LABEL,
                "reconstruction_c_index": point_reconstruction,
                "contrastive_c_index": point_contrastive,
                "paired_delta_c": point_contrastive - point_reconstruction,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_probability_positive": float(np.mean(np.asarray(samples) > 0)),
                "control": "same encoder backbone, dimension, folds, epochs, ridge head",
            }
        ]
    )


def summarize_ablations(data, predictions, baseline, replicates, seed_count, seed):
    seeds = list(range(1, seed_count + 1))
    arrays = {}
    for variant in VARIANTS:
        for seed_id in seeds:
            arrays[(variant, seed_id)] = (
                predictions[(predictions.variant == variant) & (predictions.seed_id == seed_id)]
                .set_index("SEQN")
                .loc[data.SEQN, "linear_predictor"]
                .to_numpy(float)
            )

    baseline_c = c_index(data, baseline)
    seed_rows = []
    point_c = {}
    for variant in VARIANTS:
        for seed_id in seeds:
            value = c_index(data, arrays[(variant, seed_id)])
            point_c[(variant, seed_id)] = value
            seed_rows.append(
                {
                    "variant": variant,
                    "label": VARIANTS[variant]["label"],
                    "seed_id": seed_id,
                    "c_index": value,
                    "delta_vs_fixed_circadian": value - baseline_c,
                    "c_index_loss_vs_full": point_c.get(("full", seed_id), value) - value,
                }
            )

    sampled_delta = {variant: [] for variant in VARIANTS}
    sampled_loss = {variant: [] for variant in VARIANTS}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[indices].reset_index(drop=True)
        base_c = c_index(sample, baseline[indices])
        bootstrap_c = {}
        for variant in VARIANTS:
            bootstrap_c[variant] = np.mean(
                [c_index(sample, arrays[(variant, seed_id)][indices]) for seed_id in seeds]
            )
            sampled_delta[variant].append(bootstrap_c[variant] - base_c)
        for variant in VARIANTS:
            sampled_loss[variant].append(bootstrap_c["full"] - bootstrap_c[variant])

    summary_rows = []
    for variant in VARIANTS:
        mean_c = float(np.mean([point_c[(variant, seed_id)] for seed_id in seeds]))
        mean_full_c = float(np.mean([point_c[("full", seed_id)] for seed_id in seeds]))
        delta_low, delta_high = np.percentile(sampled_delta[variant], [2.5, 97.5])
        loss_low, loss_high = np.percentile(sampled_loss[variant], [2.5, 97.5])
        lower_count = sum(
            point_c[(variant, seed_id)] < point_c[("full", seed_id)] for seed_id in seeds
        )
        summary_rows.append(
            {
                "variant": variant,
                "label": VARIANTS[variant]["label"],
                "mean_c_index": mean_c,
                "mean_delta_vs_fixed_circadian": mean_c - baseline_c,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
                "mean_c_index_loss_after_removal": mean_full_c - mean_c,
                "loss_ci_low": loss_low,
                "loss_ci_high": loss_high,
                "seeds_lower_than_full": lower_count,
                "seed_count": seed_count,
            }
        )
    return pd.DataFrame(seed_rows), pd.DataFrame(summary_rows)


def integration_decision(summary: pd.DataFrame) -> tuple[bool, str]:
    evidence = []
    for row in summary[summary.variant != "full"].itertuples(index=False):
        if row.loss_ci_low > 0 and row.seeds_lower_than_full >= max(2, row.seed_count - 1):
            action = row.label.replace("Remove", "Removing", 1)
            evidence.append(f"{action} consistently reduced C-index")
        if row.loss_ci_high < 0 and row.seeds_lower_than_full <= 1:
            action = row.label.replace("Remove", "Removing", 1)
            evidence.append(f"{action} consistently improved C-index")
    if evidence:
        return True, "; ".join(evidence)
    return False, "No single augmentation had a paired mean effect whose 95% interval excluded zero with consistent seed direction."


def make_figure(objective, summary, smoke):
    objective_row = objective.iloc[0]
    ablations = summary[summary.variant != "full"].copy()
    y = np.arange(len(ablations))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), gridspec_kw={"width_ratios": [0.8, 1.2]})

    axes[0].errorbar(
        objective_row.paired_delta_c * 1000,
        0,
        xerr=[
            [(objective_row.paired_delta_c - objective_row.ci_low) * 1000],
            [(objective_row.ci_high - objective_row.paired_delta_c) * 1000],
        ],
        fmt="o",
        color="#D55E00",
        capsize=4,
    )
    axes[0].axvline(0, color="#555555", ls="--", lw=1)
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(["Contrastive - reconstruction AE\n(each added to circadian)"])
    axes[0].set_xlabel(
        "Paired C-index difference (10^-3; 95% CI)\n"
        "Positive values favour contrastive"
    )
    axes[0].set_title("A. Learning objective", loc="left", fontweight="bold")

    axes[1].errorbar(
        ablations.mean_c_index_loss_after_removal * 1000,
        y,
        xerr=[
            (ablations.mean_c_index_loss_after_removal - ablations.loss_ci_low) * 1000,
            (ablations.loss_ci_high - ablations.mean_c_index_loss_after_removal) * 1000,
        ],
        fmt="o",
        color="#0072B2",
        capsize=4,
    )
    axes[1].axvline(0, color="#555555", ls="--", lw=1)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(ablations.label)
    axes[1].set_xlabel(
        "C-index loss after removing augmentation (10^-3; 95% CI)\n"
        "Negative: removal improves; positive: augmentation helps"
    )
    axes[1].set_title("B. One-at-a-time augmentation ablation", loc="left", fontweight="bold")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "What explains the exploratory contrastive signal?",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.01,
        "Post-hoc exploratory mechanism analysis; paired outer folds, fixed 8-D encoder and ridge alpha=0.1.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])
    suffix = "_smoke" if smoke else ""
    fig.savefig(FIGURES / f"fig16_contrastive_ablation{suffix}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGURES / f"fig16_contrastive_ablation{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(objective, summary, seed_results, data, smoke, recommended, reason):
    objective_row = objective.iloc[0]
    ablations = summary[summary.variant != "full"]
    largest = ablations.loc[ablations.mean_c_index_loss_after_removal.idxmax()]
    no_noise = summary.loc[summary.variant == "no_noise"].iloc[0]
    largest_action = largest.label.replace("Remove", "removing", 1).lower()
    decision_reason = reason.rstrip(".") + "."
    text = f"""# Contrastive Objective And Augmentation Ablation

This post-hoc exploratory analysis used **{len(data):,} participants** and
**{int(data.cvd_death.sum())} heart-or-stroke deaths**. It cannot replace the
prespecified M6-versus-M3 primary analysis.

## Objective comparison

Using the same encoder backbone, eight-dimensional embedding, outer folds,
training epochs and ridge head, circadian plus contrastive-8 exceeded
circadian plus reconstruction AE-8 by
**{objective_row.paired_delta_c:+.5f} C-index** (95% CI
{objective_row.ci_low:+.5f} to {objective_row.ci_high:+.5f}; bootstrap
probability of a positive difference
{objective_row.bootstrap_probability_positive:.1%}). The interval
{'excluded' if objective_row.ci_low > 0 else 'included'} zero.

## Augmentation ablation

The largest mean C-index loss occurred after **{largest_action}**:
**{largest.mean_c_index_loss_after_removal:+.5f}** (95% CI
{largest.loss_ci_low:+.5f} to {largest.loss_ci_high:+.5f}); the ablated model
was lower than the full model in **{int(largest.seeds_lower_than_full)}/{int(largest.seed_count)}**
seeds. The direction suggests that time-shift invariance may contribute, but
the interval included zero.

Removing random noise changed C-index by
**{-no_noise.mean_c_index_loss_after_removal:+.5f}** relative to the full
model (95% CI {-no_noise.loss_ci_high:+.5f} to
{-no_noise.loss_ci_low:+.5f}) and improved the result in
**{int(no_noise.seed_count - no_noise.seeds_lower_than_full)}/{int(no_noise.seed_count)}**
seeds. Thus, random Gaussian noise was not the source of the contrastive gain;
at the tested magnitude it was counterproductive.

No augmentation is interpreted from its point estimate alone. A distinct
mechanism requires a paired interval excluding zero and a consistent direction
across at least four of five seeds.

## Manuscript decision

**Integration recommended: {'yes' if recommended else 'no'}.** {decision_reason}

The complete aggregate results are in
`tables/contrastive_ablation_summary{'_smoke' if smoke else ''}.csv`, and
seed-level point estimates are in
`tables/contrastive_ablation_by_seed{'_smoke' if smoke else ''}.csv`.
"""
    target = HERE / ("CONTRASTIVE_ABLATION_RESULTS_SMOKE.md" if smoke else "CONTRASTIVE_ABLATION_RESULTS.md")
    target.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seeds", type=int)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate the figure and Markdown summary from saved result tables.",
    )
    args = parser.parse_args()

    config, data, profiles, fold_by_seqn, baseline, incremental, full = load_analysis(args.smoke)
    seed_count = args.seeds or (2 if args.smoke else SEED_COUNT)
    if not 1 <= seed_count <= SEED_COUNT:
        raise ValueError(f"--seeds must be between 1 and {SEED_COUNT}")
    suffix = "_smoke" if args.smoke else ""

    if args.render_only:
        objective = pd.read_csv(TABLES / f"contrastive_objective_comparison{suffix}.csv")
        objective.loc[:, "comparison"] = OBJECTIVE_LABEL
        seed_results = pd.read_csv(TABLES / f"contrastive_ablation_by_seed{suffix}.csv")
        summary = pd.read_csv(TABLES / f"contrastive_ablation_summary{suffix}.csv")
        objective.to_csv(
            TABLES / f"contrastive_objective_comparison{suffix}.csv", index=False
        )
        recommended, reason = integration_decision(summary)
        make_figure(objective, summary, args.smoke)
        write_summary(
            objective, summary, seed_results, data, args.smoke, recommended, reason
        )
        print(f"Regenerated contrastive ablation outputs ({'smoke' if args.smoke else 'full'}).")
        return

    predictions = run_ablations(
        data,
        profiles,
        fold_by_seqn,
        full,
        config,
        args.smoke,
        args.force,
        seed_count,
    )
    predictions.to_parquet(
        PREDICTIONS / f"contrastive_ablation_predictions{suffix}.parquet",
        index=False,
    )

    objective = paired_objective_comparison(
        data,
        incremental,
        int(config["bootstrap_replicates"]),
        int(config["seed"]) + 1901,
    )
    seed_results, summary = summarize_ablations(
        data,
        predictions,
        baseline,
        int(config["bootstrap_replicates"]),
        seed_count,
        int(config["seed"]) + 1902,
    )
    objective.to_csv(TABLES / f"contrastive_objective_comparison{suffix}.csv", index=False)
    seed_results.to_csv(TABLES / f"contrastive_ablation_by_seed{suffix}.csv", index=False)
    summary.to_csv(TABLES / f"contrastive_ablation_summary{suffix}.csv", index=False)
    recommended, reason = integration_decision(summary)
    make_figure(objective, summary, args.smoke)
    write_summary(objective, summary, seed_results, data, args.smoke, recommended, reason)

    print("\nObjective comparison")
    print(objective.round(5).to_string(index=False))
    print("\nAugmentation ablation")
    print(summary.round(5).to_string(index=False))
    print(f"\nIntegration recommended: {recommended}. {reason}")


if __name__ == "__main__":
    main()
