# Locked Contrastive Protocol for the Next Study

Lock date: 21 July 2026

## Purpose

The current NHANES analysis suggested that contrastive learning may preserve
activity structure useful for cardiovascular risk ranking. A post-hoc ablation
also found that Gaussian noise was counterproductive at the tested magnitude,
while removing time shifting reduced performance in all five seeds. This
protocol converts that observation into a prospectively testable hypothesis.

## Locked hypothesis

When added to PREDICT and M10/L5/RA, an eight-dimensional contrastive activity
representation trained without Gaussian noise will improve out-of-sample
cardiovascular risk prediction more than a reconstruction autoencoder and the
interpretable activity features alone.

## Locked recipe

- Input: one wear-aware 1,440-minute average-day activity profile.
- Encoder: 1,440 to 128 to 8 dimensions.
- Projector: 8 to 32 to 16 dimensions.
- Objective: contrastive loss with temperature 0.2.
- Retain six 30-minute masked blocks.
- Retain activity scaling from 0.9 to 1.1.
- Retain circular time shifting from -30 to +30 minutes.
- Do not add Gaussian noise.
- Fit the Cox correction with ridge alpha=0.1.
- Use M10, L5 and RA alongside the learned embedding.
- Fit preprocessing, representation learning, Cox coefficients, and baseline
  hazard using training participants only.

The machine-readable specification is
`analysis/NEXT_STUDY_CONTRASTIVE_CONFIG.yaml`.

## Required evaluation

The next test must use an independent or genuinely future endpoint-aligned
cohort with fatal and non-fatal cardiovascular events. Five outer folds and all
five recorded training seeds must be reported; no best seed may be selected.
The main comparison is locked contrastive-8 versus M10/L5/RA, both added to the
same PREDICT offset.

Report paired C-index change, calibration slope, calibration-in-the-large,
IPCW Brier score, and decision-curve net benefit. Clinical benefit requires
coherent absolute-risk and decision-curve evidence, not only a higher C-index.

## Interpretation boundary

The existing NHANES no-noise result is hypothesis-generating because this
recipe was chosen after examining that cohort. Re-running it on the same
participants cannot provide confirmatory evidence. Until the locked recipe is
tested on new data, it remains a future-study candidate rather than a validated
model.
