"""Verify cohort, folds, leakage boundaries, prediction schema, and determinism."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from models import CACHE, DATA, FIGURES, HERE, MODEL_IDS, PREDICTIONS, TABLES, load_config, stratified_folds


def check(rows: list[dict], name: str, condition: bool, detail: str) -> None:
    rows.append({"check": name, "passed": bool(condition), "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.smoke)
    cache_suffix = "smoke" if args.smoke else "full"
    output_suffix = "_smoke" if args.smoke else ""
    cohort = pd.read_parquet(DATA / "cohort.parquet")
    with np.load(DATA / "average_day_profiles.npz") as stored:
        seqn = stored["seqn"].astype(int)
        profiles = stored["profile"]
        observed = stored["observed_mask"].astype(bool)
    rows: list[dict] = []
    endpoint_codes = tuple(str(code) for code in config.get("endpoint", {}).get("codes", ["001"]))
    expected_events = 192 if endpoint_codes == ("001", "005") else 163
    check(rows, "canonical participant count", len(cohort) == 4341, f"observed={len(cohort)} expected=4341")
    check(rows, "canonical primary events", int(cohort.cvd_death.sum()) == expected_events, f"observed={int(cohort.cvd_death.sum())} expected={expected_events}")
    check(rows, "unique participant IDs", cohort.SEQN.is_unique, f"unique={cohort.SEQN.nunique()}")
    check(rows, "profile participant order", np.array_equal(seqn, cohort.SEQN.to_numpy(int)), "NPZ order equals cohort order")
    check(rows, "finite completed profiles", np.isfinite(profiles).all(), f"nonfinite={int((~np.isfinite(profiles)).sum())}")
    check(rows, "wear mask shape", observed.shape == profiles.shape, f"mask={observed.shape}, profile={profiles.shape}")
    check(rows, "no unobserved minute encoded as forced zero", bool(np.all(profiles[~observed] != 0)) if (~observed).any() else True, f"unobserved_minutes={int((~observed).sum())}")
    validation = pd.read_csv(TABLES / "profile_validation.csv")
    check(rows, "average-day reconstruction", validation.minimum_profile_correlation.min() >= 0.999, f"minimum_r={validation.minimum_profile_correlation.min():.6f}")
    check(rows, "MVPA reconstruction", validation.maximum_mvpa_difference.max() <= 1e-5, f"maximum_difference={validation.maximum_mvpa_difference.max():.8f}")
    assignments = pd.read_csv(TABLES / "fold_assignments.csv")
    generated = stratified_folds(cohort, 5, int(load_config()["seed"]))
    regenerated = np.full(len(cohort), -1)
    for number, indices in enumerate(generated, start=1):
        regenerated[indices] = number
    assigned = assignments.set_index("SEQN").loc[cohort.SEQN, "outer_fold"].to_numpy(int)
    check(rows, "deterministic full folds", np.array_equal(regenerated, assigned), "stored folds reproduce from seed")
    next_study_path = HERE / "NEXT_STUDY_CONTRASTIVE_CONFIG.yaml"
    next_study_ok = False
    next_study_detail = f"missing {next_study_path}"
    if next_study_path.exists():
        with next_study_path.open() as handle:
            next_study = json.load(handle)
        augmentations = next_study["augmentations"]
        next_study_ok = (
            next_study["encoder"]["embedding_dimension"] == 8
            and next_study["risk_extension"]["ridge_alpha"] == 0.1
            and augmentations["block_masking"]["enabled"]
            and augmentations["activity_scaling"]["enabled"]
            and not augmentations["gaussian_noise"]["enabled"]
            and augmentations["circular_time_shift"]["enabled"]
            and not next_study["evaluation"]["select_best_seed"]
        )
        next_study_detail = (
            "8-D, alpha=0.1; masking/scaling/shift enabled; noise disabled; "
            "best-seed selection disabled"
        )
    check(rows, "next-study contrastive recipe lock", next_study_ok, next_study_detail)

    prediction_path = PREDICTIONS / ("oof_predictions_smoke.parquet" if args.smoke else "oof_predictions.parquet")
    if prediction_path.exists():
        predictions = pd.read_parquet(prediction_path)
        participant_count = predictions.SEQN.nunique()
        expected_rows = participant_count * len(MODEL_IDS)
        check(rows, "prediction row count", len(predictions) == expected_rows, f"rows={len(predictions)} expected={expected_rows}")
        check(rows, "canonical model registry", set(predictions.model_id.unique()) == set(MODEL_IDS), f"models={sorted(predictions.model_id.unique())}")
        check(rows, "one prediction per model and participant", not predictions.duplicated(["SEQN", "model_id"]).any(), "no duplicate keys")
        check(rows, "finite linear predictors", np.isfinite(predictions.linear_predictor).all(), f"nonfinite={int((~np.isfinite(predictions.linear_predictor)).sum())}")
        check(rows, "absolute-risk range", predictions.risk_10y.between(0, 1).all(), f"range=[{predictions.risk_10y.min():.5f}, {predictions.risk_10y.max():.5f}]")
        fold_consistency = predictions.groupby("SEQN").outer_fold.nunique().max() == 1
        check(rows, "one outer fold per participant", fold_consistency, "all model rows share participant fold")
        selections = predictions.groupby(["outer_fold", "model_id"])[["selected_dimension", "selected_alpha"]].nunique(dropna=False)
        check(rows, "one selected configuration per fold-model", bool((selections <= 1).all().all()), "dimension and alpha are fold-locked")
        if not args.smoke:
            base = predictions[predictions.model_id == "M0"].set_index("SEQN").loc[cohort.SEQN]
            check(rows, "transported PREDICT LP unchanged", np.allclose(base.linear_predictor, cohort.lp_predict), "M0 OOF LP equals published PREDICT LP")
        leakage_ok = True
        leakage_details = []
        for fold_number in range(1, int(config["outer_folds"]) + 1):
            for model_id in MODEL_IDS:
                path = CACHE / f"features_{cache_suffix}_fold_{fold_number}_{model_id}.npz"
                if not path.exists():
                    leakage_ok = False; leakage_details.append(f"missing {path.name}"); continue
                with np.load(path) as stored:
                    train = stored["train_indices"].astype(int)
                    test = stored["test_indices"].astype(int)
                if np.intersect1d(train, test).size or len(np.union1d(train, test)) != participant_count:
                    leakage_ok = False; leakage_details.append(f"fold {fold_number} {model_id}")
        check(rows, "train-test index separation", leakage_ok, "; ".join(leakage_details) or "all cached train/test sets are disjoint and complete")
    else:
        check(rows, "prediction file present", False, f"missing {prediction_path}")
    if not args.smoke:
        max_mean_path = PREDICTIONS / "oof_predictions_max_mean.parquet"
        if max_mean_path.exists():
            alternative = pd.read_parquet(max_mean_path)
            check(
                rows,
                "maximum-mean prediction row count",
                len(alternative) == len(cohort) * len(MODEL_IDS),
                f"rows={len(alternative)} expected={len(cohort) * len(MODEL_IDS)}",
            )
            check(
                rows,
                "maximum-mean one prediction per participant-model",
                not alternative.duplicated(["SEQN", "model_id"]).any(),
                "no duplicate keys",
            )
            check(
                rows,
                "maximum-mean finite predictions and risks",
                np.isfinite(alternative.linear_predictor).all()
                and alternative.risk_10y.between(0, 1).all(),
                "linear predictors finite and risks within [0,1]",
            )
            alternative_folds = (
                alternative[["SEQN", "outer_fold"]]
                .drop_duplicates("SEQN")
                .set_index("SEQN")
                .loc[cohort.SEQN, "outer_fold"]
                .to_numpy(int)
            )
            check(
                rows,
                "maximum-mean uses canonical outer folds",
                np.array_equal(alternative_folds, assigned),
                "participant folds match the locked fold table",
            )
            saved_selection = pd.read_csv(TABLES / "max_mean_selections.csv")
            inner_tuning = pd.read_csv(TABLES / "nested_tuning.csv")
            selection_ok = True
            for selected in saved_selection.itertuples():
                candidates = inner_tuning[
                    (inner_tuning.outer_fold == selected.outer_fold)
                    & (inner_tuning.model_id == selected.model_id)
                ]
                best = candidates.sort_values(
                    ["mean_c_index", "dimension", "alpha"],
                    ascending=[False, True, False],
                ).iloc[0]
                selection_ok &= (
                    int(best.dimension) == int(selected.selected_dimension)
                    and np.isclose(float(best.alpha), float(selected.selected_alpha))
                )
            check(
                rows,
                "maximum-mean selections reproduce inner tuning maxima",
                selection_ok,
                "all learned-model fold selections match saved inner-fold results",
            )
        else:
            check(rows, "maximum-mean prediction file present", False, f"missing {max_mean_path}")
        for policy_name, prediction_name, source_name, table_suffix in [
            ("conservative one-SE", "oof_predictions_5y.parquet", "oof_predictions.parquet", ""),
            ("maximum inner mean", "oof_predictions_max_mean_5y.parquet", "oof_predictions_max_mean.parquet", "_max_mean"),
        ]:
            horizon_path = PREDICTIONS / prediction_name
            source_horizon_path = PREDICTIONS / source_name
            if horizon_path.exists() and source_horizon_path.exists():
                horizon_predictions = pd.read_parquet(horizon_path)
                source_predictions = pd.read_parquet(source_horizon_path)
                horizon_sorted = horizon_predictions.sort_values(["SEQN", "model_id"]).reset_index(drop=True)
                source_sorted = source_predictions.sort_values(["SEQN", "model_id"]).reset_index(drop=True)
                horizon_ok = (
                    len(horizon_sorted) == len(cohort) * len(MODEL_IDS)
                    and not horizon_sorted.duplicated(["SEQN", "model_id"]).any()
                    and set(horizon_sorted.model_id.unique()) == set(MODEL_IDS)
                    and horizon_sorted.risk_5y.between(0, 1).all()
                    and np.isfinite(horizon_sorted.linear_predictor).all()
                    and np.allclose(horizon_sorted.horizon_years, 5.0)
                    and np.array_equal(
                        horizon_sorted[["SEQN", "model_id"]].to_numpy(),
                        source_sorted[["SEQN", "model_id"]].to_numpy(),
                    )
                    and np.allclose(
                        horizon_sorted.linear_predictor,
                        source_sorted.linear_predictor,
                    )
                )
                check(
                    rows,
                    f"5-year {policy_name} prediction completeness",
                    horizon_ok,
                    f"rows={len(horizon_sorted)} expected={len(cohort) * len(MODEL_IDS)}; LPs match source",
                )
                required_horizon_tables = [
                    f"horizon_5y_clinical_metrics{table_suffix}.csv",
                    f"horizon_5y_decision_curve{table_suffix}.csv",
                    f"horizon_5y_calibration_by_decile{table_suffix}.csv",
                    f"horizon_5y_paired_primary_clinical_comparison{table_suffix}.csv",
                ]
                missing_horizon_tables = [
                    name for name in required_horizon_tables if not (TABLES / name).exists()
                ]
                check(
                    rows,
                    f"5-year {policy_name} result tables present",
                    not missing_horizon_tables,
                    "all present" if not missing_horizon_tables else f"missing={missing_horizon_tables}",
                )
            else:
                check(
                    rows,
                    f"5-year {policy_name} prediction file present",
                    False,
                    f"missing={horizon_path if not horizon_path.exists() else source_horizon_path}",
                )
        required_tables = [
            "prespecified_comparisons.csv",
            "prespecified_comparisons_max_mean.csv",
            "paired_primary_clinical_comparison.csv",
            "tuning_policy_comparison.csv",
            "subgroup_performance.csv",
        ]
        missing_tables = [name for name in required_tables if not (TABLES / name).exists()]
        check(
            rows,
            "publication comparison tables present",
            not missing_tables,
            "all present" if not missing_tables else f"missing={missing_tables}",
        )
        ai_path = PREDICTIONS / "ai_representation_predictions.parquet"
        if ai_path.exists():
            ai = pd.read_parquet(ai_path)
            expected_methods = {"B0", "B1", "P8", "AE8", "MAE8", "CON8", "RAE8"}
            check(
                rows,
                "AI experiment method registry",
                set(ai.method_id.unique()) == expected_methods,
                f"methods={sorted(ai.method_id.unique())}",
            )
            check(
                rows,
                "AI experiment prediction completeness",
                len(ai) == len(cohort) * len(expected_methods)
                and not ai.duplicated(["SEQN", "method_id"]).any(),
                f"rows={len(ai)} expected={len(cohort) * len(expected_methods)}",
            )
            check(
                rows,
                "AI experiment finite predictions and risks",
                np.isfinite(ai.linear_predictor).all() and ai.risk_10y.between(0, 1).all(),
                "linear predictors finite and risks within [0,1]",
            )
            ai_folds = (
                ai[["SEQN", "outer_fold"]]
                .drop_duplicates("SEQN")
                .set_index("SEQN")
                .loc[cohort.SEQN, "outer_fold"]
                .to_numpy(int)
            )
            check(
                rows,
                "AI experiment uses canonical outer folds",
                np.array_equal(ai_folds, assigned),
                "participant folds match the locked fold table",
            )
            ai_leakage_ok = True
            ai_details = []
            for fold_number in range(1, int(config["outer_folds"]) + 1):
                path = CACHE / f"ai_representation_full_fold_{fold_number}.npz"
                if not path.exists():
                    ai_leakage_ok = False
                    ai_details.append(f"missing {path.name}")
                    continue
                with np.load(path) as stored:
                    train = stored["train_indices"].astype(int)
                    test = stored["test_indices"].astype(int)
                if np.intersect1d(train, test).size or len(np.union1d(train, test)) != len(cohort):
                    ai_leakage_ok = False
                    ai_details.append(f"fold {fold_number}")
            check(
                rows,
                "AI experiment train-test separation",
                ai_leakage_ok,
                "; ".join(ai_details) or "all AI fold caches are disjoint and complete",
            )
            ai_tables = [
                "ai_representation_results.csv",
                "ai_representation_clinical.csv",
                "ai_representation_training.csv",
                "ai_representation_readout_r2.csv",
            ]
            missing_ai_tables = [name for name in ai_tables if not (TABLES / name).exists()]
            check(
                rows,
                "AI experiment result tables present",
                not missing_ai_tables,
                "all present" if not missing_ai_tables else f"missing={missing_ai_tables}",
            )
            readout_path = PREDICTIONS / "ai_representation_readouts.parquet"
            readout_ok = False
            readout_detail = f"missing {readout_path}"
            if readout_path.exists():
                readout = pd.read_parquet(readout_path)
                readout_ok = (
                    len(readout) == len(cohort) * 5 * 5
                    and not readout.duplicated(["SEQN", "method_id", "target"]).any()
                    and np.isfinite(readout[["observed", "predicted"]]).all().all()
                )
                readout_detail = f"rows={len(readout)} expected={len(cohort) * 25}"
            check(
                rows,
                "AI representation readout completeness",
                readout_ok,
                readout_detail,
            )
            incremental_path = PREDICTIONS / "ai_incremental_predictions.parquet"
            expected_incremental = {"B0", "B1", "CP8", "CAE8", "CMAE8", "CCON8", "CRAE8"}
            incremental_ok = False
            incremental_detail = f"missing {incremental_path}"
            if incremental_path.exists():
                incremental = pd.read_parquet(incremental_path)
                incremental_folds = (
                    incremental[["SEQN", "outer_fold"]]
                    .drop_duplicates("SEQN")
                    .set_index("SEQN")
                    .loc[cohort.SEQN, "outer_fold"]
                    .to_numpy(int)
                )
                incremental_ok = (
                    set(incremental.method_id.unique()) == expected_incremental
                    and len(incremental) == len(cohort) * len(expected_incremental)
                    and not incremental.duplicated(["SEQN", "method_id"]).any()
                    and np.isfinite(incremental.linear_predictor).all()
                    and incremental.risk_10y.between(0, 1).all()
                    and np.array_equal(incremental_folds, assigned)
                )
                incremental_detail = (
                    f"rows={len(incremental)} expected={len(cohort) * len(expected_incremental)}; "
                    f"methods={sorted(incremental.method_id.unique())}"
                )
            check(
                rows,
                "AI incremental prediction completeness",
                incremental_ok,
                incremental_detail,
            )
            incremental_tables = [
                "ai_incremental_results.csv",
                "ai_incremental_clinical.csv",
            ]
            missing_incremental = [
                name for name in incremental_tables if not (TABLES / name).exists()
            ]
            check(
                rows,
                "AI incremental result tables present",
                not missing_incremental,
                "all present" if not missing_incremental else f"missing={missing_incremental}",
            )
            stability_path = PREDICTIONS / "contrastive_seed_predictions.parquet"
            stability_ok = False
            stability_detail = f"missing {stability_path}"
            if stability_path.exists():
                stability_predictions = pd.read_parquet(stability_path)
                stability_ok = (
                    len(stability_predictions) == len(cohort) * 5
                    and set(stability_predictions.seed_id.unique()) == {1, 2, 3, 4, 5}
                    and not stability_predictions.duplicated(["SEQN", "seed_id"]).any()
                    and np.isfinite(stability_predictions.linear_predictor).all()
                )
                stability_detail = f"rows={len(stability_predictions)} expected={len(cohort) * 5}"
                incremental = pd.read_parquet(PREDICTIONS / "ai_incremental_predictions.parquet")
                seed_one = (
                    stability_predictions[stability_predictions.seed_id == 1]
                    .set_index("SEQN")
                    .loc[cohort.SEQN, "linear_predictor"]
                    .to_numpy(float)
                )
                original_contrastive = (
                    incremental[incremental.method_id == "CCON8"]
                    .set_index("SEQN")
                    .loc[cohort.SEQN, "linear_predictor"]
                    .to_numpy(float)
                )
                check(
                    rows,
                    "locked contrastive seed one reproduces incremental model",
                    np.allclose(seed_one, original_contrastive),
                    "seed-one held-out predictors equal the original CCON8 predictors",
                )
            check(rows, "contrastive seed prediction completeness", stability_ok, stability_detail)

            contrastive_transport_path = (
                PREDICTIONS / "contrastive_cycle_transport_predictions.parquet"
            )
            transport_ok = False
            transport_detail = f"missing {contrastive_transport_path}"
            if contrastive_transport_path.exists():
                contrastive_transport_predictions = pd.read_parquet(
                    contrastive_transport_path
                )
                transport_ok = (
                    len(contrastive_transport_predictions) == len(cohort) * 5
                    and set(contrastive_transport_predictions.seed_id.unique())
                    == {1, 2, 3, 4, 5}
                    and not contrastive_transport_predictions.duplicated(
                        ["SEQN", "seed_id"]
                    ).any()
                    and np.isfinite(
                        contrastive_transport_predictions[
                            ["predict_lp", "circadian_lp", "circadian_contrastive_lp"]
                        ].to_numpy(float)
                    ).all()
                    and (
                        contrastive_transport_predictions.train_cycle
                        != contrastive_transport_predictions.test_cycle
                    ).all()
                )
                transport_detail = (
                    f"rows={len(contrastive_transport_predictions)} "
                    f"expected={len(cohort) * 5}"
                )
            check(
                rows,
                "contrastive cycle-transport prediction completeness",
                transport_ok,
                transport_detail,
            )
            contrastive_tables = [
                "contrastive_seed_stability.csv",
                "contrastive_cycle_transport.csv",
            ]
            missing_contrastive = [
                name for name in contrastive_tables if not (TABLES / name).exists()
            ]
            check(
                rows,
                "locked contrastive result tables present",
                not missing_contrastive,
                "all present" if not missing_contrastive else f"missing={missing_contrastive}",
            )
            repeated_split_path = PREDICTIONS / "repeated_split_predictions.parquet"
            repeated_split_ok = False
            repeated_split_detail = f"missing {repeated_split_path}"
            if repeated_split_path.exists():
                repeated_split = pd.read_parquet(repeated_split_path)
                expected_split_seeds = {2026, 2127, 2228, 2329, 2430}
                repeated_split_ok = (
                    len(repeated_split) == len(cohort) * len(expected_split_seeds)
                    and set(repeated_split.split_seed.unique()) == expected_split_seeds
                    and not repeated_split.duplicated(["SEQN", "split_seed"]).any()
                    and np.isfinite(
                        repeated_split[["m3_lp", "m6_lp", "b1_lp", "ccon8_lp"]]
                        .to_numpy(float)
                    ).all()
                )
                fold_reproduction_ok = True
                for split_seed in expected_split_seeds:
                    generated_split = stratified_folds(cohort, 5, split_seed)
                    expected_fold = np.full(len(cohort), -1)
                    for fold_number, indices in enumerate(generated_split, start=1):
                        expected_fold[indices] = fold_number
                    observed_fold = (
                        repeated_split[repeated_split.split_seed == split_seed]
                        .set_index("SEQN")
                        .loc[cohort.SEQN, "outer_fold"]
                        .to_numpy(int)
                    )
                    fold_reproduction_ok &= np.array_equal(expected_fold, observed_fold)
                repeated_split_ok &= fold_reproduction_ok
                repeated_split_detail = (
                    f"rows={len(repeated_split)} expected={len(cohort) * 5}; "
                    f"seeds={sorted(repeated_split.split_seed.unique())}; "
                    f"folds_reproduced={fold_reproduction_ok}"
                )
            check(
                rows,
                "repeated outer-partition prediction completeness",
                repeated_split_ok,
                repeated_split_detail,
            )
            repeated_split_tables = [
                "repeated_split_stability.csv",
                "repeated_split_summary.csv",
            ]
            missing_repeated_split = [
                name for name in repeated_split_tables if not (TABLES / name).exists()
            ]
            repeated_split_figure = FIGURES / "fig17_repeated_split_stability.png"
            check(
                rows,
                "repeated outer-partition result artifacts present",
                not missing_repeated_split and repeated_split_figure.exists(),
                "all present"
                if not missing_repeated_split and repeated_split_figure.exists()
                else (
                    f"missing_tables={missing_repeated_split}; "
                    f"figure={repeated_split_figure.exists()}"
                ),
            )
            ablation_path = PREDICTIONS / "contrastive_ablation_predictions.parquet"
            ablation_ok = False
            ablation_detail = f"missing {ablation_path}"
            if ablation_path.exists():
                ablation_predictions = pd.read_parquet(ablation_path)
                expected_variants = {
                    "full", "no_mask", "no_scale", "no_noise", "no_shift"
                }
                expected_ablation_rows = len(cohort) * 5 * len(expected_variants)
                ablation_fold = (
                    ablation_predictions[["SEQN", "outer_fold"]]
                    .drop_duplicates("SEQN")
                    .set_index("SEQN")
                    .loc[cohort.SEQN, "outer_fold"]
                    .to_numpy(int)
                )
                ablation_ok = (
                    len(ablation_predictions) == expected_ablation_rows
                    and set(ablation_predictions.variant.unique()) == expected_variants
                    and set(ablation_predictions.seed_id.unique()) == {1, 2, 3, 4, 5}
                    and not ablation_predictions.duplicated(
                        ["SEQN", "seed_id", "variant"]
                    ).any()
                    and np.isfinite(ablation_predictions.linear_predictor).all()
                    and np.array_equal(ablation_fold, assigned)
                )
                ablation_detail = (
                    f"rows={len(ablation_predictions)} expected={expected_ablation_rows}; "
                    f"variants={sorted(ablation_predictions.variant.unique())}"
                )
            check(
                rows,
                "contrastive ablation prediction completeness",
                ablation_ok,
                ablation_detail,
            )
            ablation_tables = [
                "contrastive_objective_comparison.csv",
                "contrastive_ablation_by_seed.csv",
                "contrastive_ablation_summary.csv",
            ]
            missing_ablation = [
                name for name in ablation_tables if not (TABLES / name).exists()
            ]
            ablation_figure = FIGURES / "fig16_contrastive_ablation.png"
            check(
                rows,
                "contrastive ablation result artifacts present",
                not missing_ablation and ablation_figure.exists(),
                "all present" if not missing_ablation and ablation_figure.exists()
                else f"missing_tables={missing_ablation}; figure={ablation_figure.exists()}",
            )
        else:
            check(rows, "AI experiment prediction file present", False, f"missing {ai_path}")
        wrist_cohort_path = DATA / "wrist_replication_cohort.parquet"
        wrist_predictions_path = PREDICTIONS / "wrist_mims_replication.parquet"
        if wrist_cohort_path.exists() and wrist_predictions_path.exists():
            wrist_cohort = pd.read_parquet(wrist_cohort_path)
            wrist_predictions = pd.read_parquet(wrist_predictions_path)
            wrist_ok = (
                len(wrist_cohort) == 5873
                and int(wrist_cohort.cvd_death.sum()) == 80
                and len(wrist_predictions) == len(wrist_cohort)
                and not wrist_predictions.SEQN.duplicated().any()
                and np.isfinite(
                    wrist_predictions[["predict_lp", "predict_wrist_lp"]].to_numpy(float)
                ).all()
                and set(wrist_cohort.endpoint.unique()) == {"heart_or_stroke_death"}
            )
            check(
                rows,
                "wrist cross-device prediction completeness",
                wrist_ok,
                f"participants={len(wrist_cohort)} events={int(wrist_cohort.cvd_death.sum())}",
            )
            wrist_tables = [
                "wrist_cross_device_replication.csv",
                "wrist_replication_discrimination.csv",
                "wrist_replication_mapping_sensitivity.csv",
            ]
            missing_wrist_tables = [name for name in wrist_tables if not (TABLES / name).exists()]
            wrist_figure = FIGURES / "fig15_wrist_cross_device_replication.png"
            check(
                rows,
                "wrist cross-device result artifacts present",
                not missing_wrist_tables and wrist_figure.exists(),
                "all present" if not missing_wrist_tables and wrist_figure.exists()
                else f"missing_tables={missing_wrist_tables}; figure={wrist_figure.exists()}",
            )
            mapping = pd.read_csv(TABLES / "wrist_replication_mapping_sensitivity.csv")
            check(
                rows,
                "wrist direct-ethnicity sensitivity cohort",
                len(mapping) == 1
                and int(mapping.iloc[0].participants) == 3151
                and int(mapping.iloc[0].events) == 37,
                f"participants={int(mapping.iloc[0].participants)} events={int(mapping.iloc[0].events)}",
            )
        else:
            check(
                rows,
                "wrist cross-device prediction files present",
                False,
                f"cohort={wrist_cohort_path.exists()} predictions={wrist_predictions_path.exists()}",
            )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / f"verification{output_suffix}.csv", index=False)
    print(result.to_string(index=False))
    failed = result[~result.passed]
    if len(failed):
        raise AssertionError(f"{len(failed)} verification checks failed")


if __name__ == "__main__":
    main()
