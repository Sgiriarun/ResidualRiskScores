#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export MPLCONFIGDIR=/tmp/matplotlib-average-day-cvd
cd "$ROOT"

python analysis/00_build_weekly_data.py
python analysis/01_build_cohort.py
python analysis/02_run_nested_models.py
python analysis/03_evaluate_models.py
python analysis/04_run_sensitivities.py
python analysis/05_run_explainability.py
python analysis/07_dimension_sensitivity.py
python analysis/08_make_prespecified_comparisons.py
python analysis/09_run_max_mean_tuning.py
python analysis/08_make_prespecified_comparisons.py --policy maximum_inner_mean
python analysis/10_run_comparison_checks.py
python analysis/12_run_ai_representation_experiment.py
python analysis/13_run_ai_incremental_experiment.py
python analysis/14_validate_locked_contrastive.py
python analysis/15_run_reduced_clinical_experiment.py
python analysis/17_run_horizon_sensitivity.py
python analysis/18_run_wrist_replication.py
python analysis/19_run_contrastive_ablation.py
python analysis/20_run_repeated_split_stability.py
python analysis/16_make_manuscript_tables.py
python analysis/validation.py
python results/make_journal_figures.py

printf 'Pipeline completed. Generated outputs are under analysis/; frozen publication outputs are under results/.\n'
