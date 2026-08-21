# Analysis Freeze: Average-Day Wearable CVD Risk Study

Freeze date: 20 July 2026

This document fixes the analysis hierarchy used for manuscript writing. No
additional model, dimension, penalty, augmentation, outcome, threshold or
subgroup result will replace the analyses below after inspection of its test
performance. New methods may be studied later, but they must be labelled as a
new exploratory study and require independent validation.

## Fixed Study Population And Outcome

- Data: NHANES 2003-2004 and 2005-2006.
- Participants: 4,341 adults.
- Events: 192 fatal heart-or-cerebrovascular deaths.
- Endpoint codes: `UCOD_LEADING=001` or `005`.
- Endpoint wording: fatal heart-or-cerebrovascular mortality proxy.
- Clinical baseline: published PREDICT-BMI coefficients retained as a fixed
  linear predictor.
- Accelerometry: one 1,440-minute average-day hip-accelerometer profile per
  participant; non-wear and missing minutes are not treated as zero activity.

## Fixed Validation Design

- Random seed: 2026.
- Five event-and-cycle-stratified outer folds.
- Three inner folds for primary representation and ridge selection.
- The same participant folds are used for every model.
- Scaling, representation learning, tuning, Cox correction and baseline-hazard
  estimation use outer-training participants only.
- Primary uncertainty: 1,000 paired participant bootstrap samples.

## Fixed Model Ladder

| ID | Model | Role |
|---|---|---|
| M0 | Transported PREDICT linear predictor | Clinical baseline |
| M1 | PREDICT + volume | Simple activity baseline |
| M2 | PREDICT + MVPA | Activity-intensity baseline |
| M3 | PREDICT + M10/L5/RA | Primary interpretable comparator |
| M4 | PREDICT + nested-selected PCA | Secondary linear representation |
| M5 | PREDICT + nested-selected AE | Secondary nonlinear representation |
| M6 | PREDICT + M10/L5/RA + nested-selected AE | Primary augmented model |

The primary comparison is M6 versus M3. Candidate dimensions are 2, 4, 8, 16
and 32. Candidate ridge penalties are 0.001, 0.01, 0.1 and 1.0. The primary
policy selects the smallest dimension within one standard error of the best
inner C-index and then the strongest eligible ridge penalty. Maximum-inner-mean
selection is a tuning sensitivity, not an alternative primary result.

## Fixed AI Methods Analysis

The AI representation-objective benchmark is exploratory. PCA, reconstruction
AE, masked AE, contrastive learning and risk-aware masked AE use the same outer
folds, dimension 8 and ridge alpha 0.1. Contrastive-8 is the locked exploratory
candidate because it had the strongest observed signal. It is evaluated across
all five recorded training seeds and both C-to-D and D-to-C cycle-held-out
directions. No best seed is selected.

The contrastive result must not replace the nested M6-versus-M3 primary test.
It must be described as exploratory because its configuration was fixed after
earlier representation analyses in the same cohort.

## Fixed Evaluation Hierarchy

1. Primary: paired C-index difference for M6 versus M3 under nested one-SE
   selection.
2. Secondary: the complete M0-M6 discrimination ladder and maximum-inner-mean
   tuning sensitivity.
3. Exploratory AI: representation-objective benchmark, contrastive addition
   beyond circadian features, seed stability and cycle-held-out transport.
4. Absolute-risk checks: calibration slope, calibration-in-the-large, IPCW
   Brier score and decision curves.
5. Explainability: cross-validated known-feature readout, latent correlations
   and out-of-fold correction phenotypes.

Ten-year absolute-risk analyses remain the original analysis. Five-year
absolute-risk analyses are a prespecified alignment sensitivity because the
published PREDICT score uses a five-year horizon. Discrimination is unchanged
because both horizons use the same out-of-fold linear predictors. The 5%, 10%
and 15% thresholds are analytic checks only; they are not validated treatment
thresholds for the NHANES mortality proxy.

## Frozen Conclusions

- The primary nested AE comparison has a positive point estimate, but its
  confidence interval crosses zero; confirmatory incremental discrimination is
  not established.
- Contrastive-8 gives a small, stable exploratory ranking signal beyond
  circadian features.
- Five-year and ten-year absolute-risk analyses do not establish coherent
  improvement in prediction error or threshold-based clinical utility.
- Cycle-held-out transport is an internal transport check, not independent
  external validation.
- The study supports a leakage-safe AI methods contribution, not clinical
  deployment, replacement of PREDICT or validation of original PREDICT absolute
  risk.

## Approved Supplementary Cross-device Check

NHANES 2011-2014 wrist MIMS is included only as an exploratory replication of
the one-variable activity-volume association. It uses no PCA, autoencoder,
contrastive representation or clinical-utility analysis. Its purpose is to
test directional consistency under a later cycle, wrist placement and MIMS
measurement scale. It cannot replace any M0-M6 result and is not external
validation of the learned hip representation.

## Approved Post-hoc Contrastive Mechanism Analysis

A supplementary mechanism analysis compares contrastive-8 with reconstruction
AE-8 using the same encoder backbone, folds, embedding dimension, training
epochs and ridge head. It then removes block masking, activity scaling, random
noise and time shifting one at a time across five recorded seeds. Retained
augmentations use paired random draws so that each comparison changes only the
named augmentation.

This analysis is exploratory and was performed after inspection of the locked
contrastive result. It may explain which training choices deserve prospective
testing, but it cannot replace the primary M6-versus-M3 comparison or promote a
post-hoc no-noise variant to the selected model.

## Prospectively Locked Next-study Recipe

The next independent study will test the existing eight-dimensional
contrastive architecture with block masking, activity scaling and circular time
shifting retained, and Gaussian noise disabled. Ridge alpha remains 0.1. This
changes only the augmentation that showed a distinct adverse post-hoc signal.
The complete frozen specification is in
`analysis/NEXT_STUDY_CONTRASTIVE_CONFIG.yaml` and its interpretation rules are
in `docs/NEXT_STUDY_CONTRASTIVE_PROTOCOL.md`.

This prospective lock does not alter any current-paper model or conclusion.
The existing NHANES no-noise estimate cannot serve as confirmation of a recipe
selected from the same data.

## Canonical Evidence Files

- Primary predictions: `predictions/oof_predictions.parquet`
- Five-year predictions: `predictions/oof_predictions_5y.parquet`
- AI objective predictions: `predictions/ai_representation_predictions.parquet`
- AI incremental predictions: `predictions/ai_incremental_predictions.parquet`
- Contrastive seed predictions: `predictions/contrastive_seed_predictions.parquet`
- Contrastive ablation predictions:
  `predictions/contrastive_ablation_predictions.parquet`
- Primary results: `tables/primary_results.csv`
- Ten-year clinical metrics: `tables/clinical_metrics.csv`
- Five-year clinical metrics: `tables/horizon_5y_clinical_metrics.csv`
- Five-year paired comparison:
  `tables/horizon_5y_paired_primary_clinical_comparison.csv`

All manuscript numbers must come from these generated files. Manual replacement
of a result by a more favourable exploratory estimate is outside this freeze.
