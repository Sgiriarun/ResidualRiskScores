# Study protocol

## Research question

Does a participant's average 24-hour hip-accelerometer profile add prognostic
information to a transported PREDICT linear predictor, and does a learned
representation add information beyond M10, L5, and relative amplitude?

The single primary contrast is PREDICT LP + M10/L5/RA + autoencoder (M6) versus
PREDICT LP + M10/L5/RA (M3). Other model contrasts are secondary or exploratory
as specified in `PAPER_ANALYSIS_FLOW.md`.

## Cohort and endpoint

The analysis pools NHANES 2003-2004 and 2005-2006 participants aged 30-79 years
without recorded prior cardiovascular disease who have the clinical variables
required by PREDICT, linked mortality follow-up, and at least four valid
accelerometer days. The primary event is fatal heart-or-cerebrovascular
mortality, using diseases of heart (`UCOD_LEADING=001`) and cerebrovascular
diseases (`UCOD_LEADING=005`). Other deaths are censored in the primary
cause-specific Cox analysis. This is a public-mortality proxy for PREDICT CVD
events, not the original fatal/non-fatal PREDICT endpoint.

Reliable accelerometer minutes require `PAXSTAT=1` and `PAXCAL=1`. A run of at
least 60 consecutive zero-count minutes is non-wear. A valid day has at least
600 wear minutes. Average-day values are means over observed wear at each clock
minute. Clock minutes never observed for a participant remain missing during
reconstruction and are cyclically interpolated before representation learning;
they are not interpreted as zero activity.

## Representations

- Total activity volume: mean counts per observed wear minute.
- MVPA: mean daily minutes at or above 2020 counts/min.
- Circadian: M10, L5, and relative amplitude from the completed average day.
- PCA: linear components learned only from the relevant training partition.
- Autoencoder: reconstruction representation learned without outcomes and only
  from the relevant training partition.
- Circadian plus autoencoder: direct test of incremental AI information beyond
  the strongest interpretable daily-pattern comparator.

## Validation and selection

Five outer folds are stratified jointly by NHANES cycle and primary event. Each
outer training set contains three similarly stratified inner folds. Dimension
and ridge penalty are selected by mean inner-fold C-index. Among configurations
within one standard error of the best mean, the smallest dimension is selected;
within that dimension, the largest eligible ridge penalty is selected.

All profile scaling, missing-value imputation, PCA/autoencoder fitting, feature
standardization, Cox correction, and Breslow baseline hazard estimation occur
without access to the outer-test participants. Absolute-risk metrics therefore
use strictly out-of-fold 10-year predictions.

## Outcomes and interpretation

The published PREDICT coefficients are used as a fixed clinical linear-predictor
offset. Because the endpoint and horizon differ from published PREDICT, the
fold-specific NHANES baseline hazard is re-estimated; this is not a direct
external validation of original PREDICT absolute risk.

The primary metric is Harrell's C-index. Paired changes are reported for the
locked contrasts in `tables/prespecified_comparisons.csv` using 1,000 participant
bootstrap resamples. Secondary metrics are
calibration slope, mean predicted minus Kaplan-Meier observed 10-year risk,
IPCW Brier score, and decision-curve net benefit. The 5%, 10%, and 15%
thresholds are exploratory for this transported mortality endpoint and are not
presented as validated treatment thresholds.

Heart-only mortality, direct PREDICT ethnicity mapping, C-to-D/D-to-C
transport, and competing non-primary mortality are sensitivity analyses. Latent
dimension correlations from a full-cohort descriptive fit are separated from
cross-validated readout R-squared and correction-phenotype analyses.
