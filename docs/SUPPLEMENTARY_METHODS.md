# Supplementary Methods

This supplement gives the implementation-level details behind
`JOURNAL_SUBMISSION_DRAFT.md`. It is written so that a reviewer can audit the
analysis logic without reading the full source code.

## S1. Endpoint And Follow-up

The primary endpoint is fatal heart-or-cerebrovascular mortality using public
NHANES linked-mortality codes:

| Code | Meaning |
|---|---|
| `001` | Diseases of heart |
| `005` | Cerebrovascular diseases |

Other deaths are treated as censored in the primary cause-specific Cox
analysis. This endpoint is a mortality proxy and does not equal the original
PREDICT fatal/non-fatal cardiovascular endpoint.

The linked-mortality files used were:

| NHANES cycle | Public linked-mortality file |
|---|---|
| 2003-2004 | `NHANES_2003_2004_MORT_2019_PUBLIC.dat` |
| 2005-2006 | `NHANES_2005_2006_MORT_2019_PUBLIC.dat` |

The original analysis used a 10-year prediction horizon. A completed 5-year
horizon sensitivity aligns the absolute-risk checks with the published PREDICT
horizon. It retains the same out-of-fold linear predictors and recomputes only
fold-specific baseline hazards and horizon-dependent absolute-risk metrics.

### S1.1 Cohort Construction And Event Distribution

Participants were included only if they had linked mortality follow-up, valid
accelerometry, observed systolic blood pressure and observed total-to-HDL
cholesterol ratio. Missing BMI was retained through the published PREDICT
missing-BMI category. Missing binary questionnaire or medication indicators
were encoded as absent in the implemented mapping. Atrial fibrillation, family
history and antithrombotic treatment were unavailable in cycles C/D and were
set to zero.

Non-Hispanic White participants used the PREDICT European reference category.
The older NHANES ethnicity variable had no Asian category, while PREDICT had no
direct Black or Hispanic category. Black and Hispanic participants therefore
used the European reference coefficient in the primary transport analysis.
The direct-mapping sensitivity was restricted to non-Hispanic White
participants. These choices are transport assumptions and are not equivalent
to refitting or validating ethnicity-specific PREDICT coefficients in NHANES.

The generated cohort flow was:

| Stage | Participants | Events |
|---|---:|---:|
| Valid accelerometer profiles | 4,341 | -- |
| Clinical and mortality eligible for the primary endpoint | 4,949 | 216 |
| Canonical merged cohort | 4,341 | 192 |

Primary events by NHANES cycle were:

| Cycle | Participants | Heart-or-stroke deaths |
|---|---:|---:|
| C, 2003-2004 | 2,167 | 109 |
| D, 2005-2006 | 2,174 | 83 |

The deterministic five outer folds were stratified by cycle and endpoint:

| Outer fold | Cycle C participants/events | Cycle D participants/events |
|---|---:|---:|
| 1 | 433 / 22 | 434 / 16 |
| 2 | 433 / 21 | 434 / 16 |
| 3 | 434 / 22 | 435 / 17 |
| 4 | 434 / 22 | 436 / 17 |
| 5 | 433 / 22 | 435 / 17 |

## S2. Average-day Accelerometer Reconstruction

For participant `i`, day `d`, and clock minute `m`, let:

```text
c_i,d,m = observed accelerometer count
w_i,d,m = 1 if the minute is reliable wear, otherwise 0
```

Reliable wear requires valid device status and calibration indicators. Non-wear
is defined as at least 60 consecutive zero-count minutes. A valid day requires
at least 600 wear minutes, and a participant requires at least four valid days.

The average-day profile is:

```text
z_i,m = sum_d c_i,d,m * w_i,d,m / sum_d w_i,d,m
```

for clock minutes with at least one observed wear minute. If a clock minute is
never observed for a participant, it is not set to zero. It is cyclically
interpolated before profile-level representation learning.

This distinction matters: zero activity is a real observation, while missing or
non-wear is absence of observation.

## S3. Interpretable Activity Features

The manuscript uses the following feature definitions:

| Feature | Definition |
|---|---|
| Volume | Mean counts per observed wear minute |
| MVPA | Average daily minutes with counts/min >= 2020 |
| M10 | Mean activity in the most active continuous 10-hour window of the average day |
| L5 | Mean activity in the least active continuous 5-hour window of the average day |
| RA | Relative amplitude, `(M10 - L5) / (M10 + L5)` |

M10, L5 and RA are treated as interpretable daily-rhythm features. They are not
used to claim circadian physiology directly; they describe rest-activity
patterning in the average-day profile.

## S4. Frozen PREDICT Offset And Wearable Cox Correction

For participant `i`, let:

```text
LP_PREDICT_i = published PREDICT-BMI clinical linear predictor
r_i          = activity feature vector or learned representation
```

The fitted model is:

```text
h_i(t) = h_0(t) * exp(LP_PREDICT_i + beta' r_i)
```

The PREDICT coefficients are not refitted in NHANES. The only fitted component
is the activity correction `beta' r_i`, plus a fold-specific baseline hazard for
the NHANES mortality proxy endpoint.

This is why the paper says "transported PREDICT linear predictor" rather than
"validated PREDICT absolute risk".

## S5. Baseline-hazard Recalibration

Within each outer-training fold, the Breslow cumulative baseline hazard is
estimated from training participants only.

Let:

```text
eta_i = fitted linear predictor for participant i
center = mean(eta_i) in the training fold
```

The centered risk weight is:

```text
weight_i = exp(eta_i - center)
```

At each observed event time `t_j` up to the horizon:

```text
dH_0(t_j) = number of events at t_j / sum_{k at risk at t_j} weight_k
```

The cumulative baseline hazard at the horizon is:

```text
H_0(T) = sum_{t_j <= T} dH_0(t_j)
```

For an outer-test participant with linear predictor `eta_test`, predicted
absolute risk is:

```text
risk_test(T) = 1 - exp[-H_0(T) * exp(eta_test - center)]
```

No outer-test participant contributes to `H_0(T)` or `center`.

## S6. Canonical Nested Validation Algorithm

The primary analysis uses five event-and-cycle-stratified outer folds and three
inner folds.

```text
For each outer fold:
    Split participants into outer training and outer test.

    For each model M0-M6:
        If the model has no learned representation:
            Fit feature scaling and Cox correction on outer training only.

        If the model has PCA or AE representation:
            For each candidate dimension in {2, 4, 8, 16, 32}:
                For each ridge alpha in {0.001, 0.01, 0.1, 1.0}:
                    Evaluate mean inner-fold C-index.

            Select the smallest dimension within one standard error
            of the best mean inner C-index.

            Within that dimension, select the strongest eligible ridge penalty.

            Refit scaling and representation learning on the full outer
            training fold only.

        Fit Cox correction on outer training only.
        Estimate baseline hazard on outer training only.
        Predict linear predictor and 10-year risk once for outer-test participants.
```

The final evaluation concatenates out-of-fold predictions from all outer folds.

## S7. Leakage Controls

The following operations are fitted separately inside each outer-training fold:

| Operation | Test participants excluded? |
|---|---|
| Profile scaling | Yes |
| Feature standardization | Yes |
| PCA basis | Yes |
| Autoencoder training | Yes |
| Contrastive encoder training | Yes |
| Inner-fold hyperparameter selection | Yes |
| Cox correction coefficients | Yes |
| Baseline hazard estimation | Yes |

The validation script checks participant counts, event counts, fold assignment,
train-test separation, prediction completeness, finite linear predictors,
absolute-risk range and reproducibility of the locked contrastive seed-one
predictions.

## S8. Harrell C-index

The C-index estimates whether participants who experience the endpoint earlier
receive higher predicted risk scores than comparable participants who survive
longer.

For comparable pairs `(i, j)`, the C-index is:

```text
C = concordant pairs / comparable pairs
```

In this study, the primary C-index is calculated from out-of-fold linear
predictors. Paired changes use the same participants under two models.

## S9. Participant-level Bootstrap

Uncertainty intervals use 1,000 participant-level bootstrap samples.

```text
For b = 1 to 1000:
    Sample N participants with replacement.
    Compute metric under model A.
    Compute metric under model B on the same sampled participants.
    Store paired difference metric(A) - metric(B).

95% CI = 2.5th and 97.5th percentiles of stored paired differences.
```

This preserves the paired nature of model comparisons.

## S10. Calibration Metrics

### Calibration Slope

Calibration slope is estimated by fitting a Cox model with the model linear
predictor as the only covariate:

```text
h_i(t) = h_0(t) * exp(gamma * eta_i)
```

The ideal slope is 1.

- Slope > 1 means predictions may be too compressed.
- Slope < 1 means predictions may be too extreme.

The clinical comparison table reports absolute distance from 1:

```text
calibration_slope_distance = abs(slope - 1)
```

Lower is better.

### Calibration-in-the-large

The analysis compares mean predicted 10-year risk with Kaplan-Meier observed
10-year risk:

```text
calibration_in_large = mean(predicted risk) - observed KM risk
```

The clinical comparison table reports absolute error. Lower is better.

### Calibration By Decile

Participants are divided into 10 groups by predicted risk. For each decile, the
mean predicted risk is plotted against the Kaplan-Meier observed 10-year event
probability.

## S11. IPCW Brier Score

The Brier score is the mean squared prediction error at the selected 5- or
10-year horizon, adjusted for right censoring using
inverse-probability-of-censoring weights.

Let `p_i` be predicted risk at the selected horizon. Let `G(t)` be the censoring survival
probability. The implementation uses:

```text
For participants with event by T:
    contribution = (1 - p_i)^2 / G(time_i)

For participants known to survive beyond T:
    contribution = p_i^2 / G(T)

Brier(T) = mean(contributions)
```

Lower values indicate better absolute-risk prediction error.

## S12. Decision-curve Net Benefit

For threshold `q`, participants with predicted risk at least `q` are considered
"high risk" for the decision rule.

The false-positive penalty is:

```text
penalty = q / (1 - q)
```

The implementation estimates the event probability among treated participants
using Kaplan-Meier at the selected horizon:

```text
NB(q) = treated_fraction * [event_probability_treated
        - (1 - event_probability_treated) * penalty]
```

This is equivalent to balancing true-positive benefit against false-positive
harm at the chosen threshold. Higher net benefit is better, but the 5%, 10%
and 15% thresholds are exploratory here because they are not validated
treatment thresholds for the NHANES mortality proxy endpoint.

For the five-year sensitivity, observed mortality risk was below 1%. The 10%
and 15% analytic thresholds therefore classified almost nobody and were not
interpreted as evidence for or against clinical utility.

## S13. AI Representation Objectives

### PCA

PCA is a linear compression baseline. It asks whether the average-day profile
contains useful low-dimensional structure without nonlinear learning.

### Reconstruction Autoencoder

The reconstruction autoencoder learns an embedding that reconstructs the full
standardized average-day profile. Its objective is unsupervised reconstruction.

### Masked Autoencoder

The masked autoencoder masks contiguous clock-time blocks and reconstructs the
missing sections. Its objective encourages the embedding to capture profile
context rather than copy each minute.

### Contrastive Encoder

The contrastive encoder creates two perturbed views of the same participant's
average-day profile. Perturbations include block masking, activity scaling,
noise and small circular time shifts. The objective pulls same-participant
views together and separates different participants.

### Risk-aware Masked Autoencoder

The risk-aware masked autoencoder adds an outcome-aware Cox component inside
the training fold. Because it uses outcome information during representation
learning, it is exploratory and not part of the conservative primary model.

## S14. Explainability Analyses

### Readout R2

Known activity features are predicted from all embedding dimensions using
training-fold embeddings, then evaluated on held-out participants. The resulting
cross-validated R2 measures how much known activity structure is retained in
the embedding.

### Latent-feature Correlations

Pearson correlations between individual latent dimensions and known activity
features are descriptive. A single coordinate should not be named as a clinical
phenotype unless the correlation pattern is strong and stable.

### Correction Phenotypes

Participants are ranked by their out-of-fold learned risk correction. The top
10% and bottom 10% are compared by average-day profile and known activity
features. This explains which activity patterns receive larger learned
corrections; it does not establish causality.

## S15. Controlled Contrastive-objective And Augmentation Ablation

The post-hoc objective comparison used the same circadian comparator, encoder
backbone, eight latent dimensions, outer folds, training epochs and ridge
alpha=0.1 for reconstruction AE-8 and contrastive-8. The only intended
difference was the representation objective. Participant-level paired
bootstrap samples quantified the C-index difference between the two combined
models.

The augmentation analysis compared the full contrastive recipe with four
one-at-a-time removals: contiguous block masking, activity scaling, Gaussian
noise and circular time shifting. Each variant used the same five outer folds
and five training seeds. All random values were drawn in the same order before
an augmentation was enabled or disabled, preserving paired randomness for the
augmentations retained in both models.

For each seed, the reported quantity was:

```text
C-index loss after removal = C-index(full recipe) - C-index(ablated recipe)
```

A positive value means the removed augmentation helped; a negative value means
that removing it improved performance. The primary uncertainty interval was a
paired participant bootstrap applied to the across-seed mean prediction. An
augmentation was considered to have a distinct mechanism signal only when the
paired interval excluded zero and at least four of five seeds agreed in
direction. These post-hoc variants were not eligible to replace the locked
contrastive model.

## S16. Exploratory Cross-device Wrist Replication

NHANES 2011-2012 and 2013-2014 used an ActiGraph GT3X+ placed on the
non-dominant wrist. The day-summary variable `PAXMTSD` is total triaxial
activity in monitor-independent movement summary (MIMS) units. Days required
at least 600 valid minutes, and participants required at least four valid days.
Mean daily MIMS was the only wrist predictor.

The wrist cohort used the same age range, prior-CVD exclusion, clinical-variable
requirements, frozen PREDICT-BMI linear predictor and fatal
heart-or-cerebrovascular mortality codes (`001+005`) as the primary study. The
cohort contained 5,873 participants and 80 events. Because of this event count,
no PCA, autoencoder, contrastive model, nonlinear transform or decision-curve
analysis was fitted.

MIMS was standardised within each cycle. A common activity coefficient was
estimated using the PREDICT linear predictor as an offset and cycle-stratified
baseline hazards. Five event-and-cycle-stratified outer folds produced held-out
linear predictors. C-index was calculated separately within cycles G and H,
then summarised using event-count weights; cross-cycle participant pairs were
not treated as comparable when baseline hazards differed. Paired confidence
intervals used 1,000 participant bootstrap samples within cycle.

The primary wrist cohort used the same ethnicity mapping as the main study. A
sensitivity analysis restricted the cohort to White and Asian participants,
the NHANES groups with directly available PREDICT categories. This restriction
contained 3,151 participants and 37 events.

The cross-device forest plot compares PREDICT-offset hazard ratios per 1 SD
higher activity. It compares association direction only. Hip counts and wrist
MIMS are not numerically interchangeable, and the learned hip representation
was not applied to the wrist cohort.

## S17. Claim Rules

The paper uses the following interpretation rules:

| Evidence | Safe claim |
|---|---|
| Higher point estimate only | Small uncertain ranking change |
| Paired C-index CI above zero | Evidence of improved discrimination |
| Calibration/Brier/net benefit directionally coherent | Possible absolute-risk or clinical-utility improvement |
| Net benefit at one exploratory threshold only | Hypothesis-generating, not clinical utility proof |
| Seed and cycle direction consistent | Computational stability, not independent external validation |

## S18. Key Reproducibility Files

| Purpose | File |
|---|---|
| Build cohort and average-day profiles | `01_build_cohort.py` |
| Run canonical nested models | `02_run_nested_models.py` |
| Evaluate primary metrics | `03_evaluate_models.py` |
| Run sensitivity analyses | `04_run_sensitivities.py` |
| Explainability analyses | `05_run_explainability.py` |
| Generate journal figures | `results/make_journal_figures.py` |
| AI representation benchmark | `12_run_ai_representation_experiment.py` |
| AI beyond circadian | `13_run_ai_incremental_experiment.py` |
| Locked contrastive validation | `14_validate_locked_contrastive.py` |
| Contrastive augmentation ablation | `19_run_contrastive_ablation.py` |
| Locked next-study contrastive recipe | `analysis/NEXT_STUDY_CONTRASTIVE_CONFIG.yaml` |
| Generate manuscript tables | `16_make_manuscript_tables.py` |
| Cross-device wrist replication | `18_run_wrist_replication.py` |
| Validation checks | `validation.py` |

The central prediction file for the canonical analysis is
`predictions/oof_predictions.parquet`.
The supplementary wrist predictions are stored separately in
`predictions/wrist_mims_replication.parquet`.
