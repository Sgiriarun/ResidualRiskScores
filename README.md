# Average-Day Accelerometry and Cardiovascular Risk

This release contains the code and aggregate evidence for evaluating whether a
wear-aware 24-hour hip-accelerometer profile adds cardiovascular risk
information beyond a transported PREDICT score.

The study uses public NHANES 2003--2006 examination, accelerometry, and linked
mortality data. Published PREDICT-BMI coefficients remain fixed. Activity
summaries and learned representations are permitted to estimate only an
additive Cox risk correction.

## Scientific boundary

- Primary endpoint: fatal heart-or-cerebrovascular mortality proxy
  (UCOD_LEADING codes 001 and 005).
- Primary comparison: PREDICT plus daily-rhythm features and a nested
  autoencoder versus PREDICT plus daily-rhythm features.
- Main result: the primary paired C-index interval included zero.
- Exploratory result: contrastive representations produced the strongest and
  most stable point estimates, but clinical utility was not established.
- This repository supports research reproducibility; it does not provide a
  deployable clinical risk calculator.

## Folder structure

~~~text
analysis/     Cohort construction, modelling, evaluation, and validation
support/      NHANES clinical loader and published PREDICT implementation
results/      Frozen aggregate tables, publication figures, and figure code
manuscript/   Journal manuscript and supplementary LaTeX sources
docs/         Frozen protocol, analysis hierarchy, methods, and reporting checklist
~~~

Raw data, model caches, and participant-level predictions are intentionally
excluded.

## Reproduce the study

Python 3.13.12 was used for the frozen run.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run_pipeline.sh
~~~

The first stage downloads the public NHANES hip-accelerometry archives and
retains participants with at least four valid days. A valid day requires at
least 600 reliable wear minutes; non-wear is defined as at least 60 consecutive
zero-count minutes. Missing and non-wear minutes are never entered as zero
activity.

The complete pipeline is computationally intensive. Individual numbered scripts
may be run separately after their required upstream outputs exist.

## Analysis map

| Stage | Files | Purpose |
|---|---|---|
| Data | 00, 01 | Build weekly wear masks, average-day profiles, clinical cohort, and folds |
| Primary | 02, 03, 08 | Nested model ladder, out-of-fold metrics, and paired contrasts |
| Robustness | 04, 07, 09, 10, 17, 20 | Endpoint, tuning, horizon, subgroup, and partition checks |
| Interpretation | 05 | Held-out feature readout and correction phenotypes |
| Representation objectives | 12, 13, 14, 19 | Reconstruction, masked, contrastive, incremental, seed, transport, and ablation analyses |
| Additional evidence | 15, 18 | Reduced-clinical-model and later wrist-cohort checks |
| Reporting | 16, validation.py, results/make_journal_figures.py | Manuscript tables, integrity checks, and journal figures |

**models.py**, **metrics.py**, and **ai_representation_models.py** contain the
shared survival, evaluation, and representation implementations.

The prospectively locked follow-up recipe is recorded in
**analysis/NEXT_STUDY_CONTRASTIVE_CONFIG.yaml** and explained in
**docs/NEXT_STUDY_CONTRASTIVE_PROTOCOL.md**. It is not a confirmatory result
from the current cohort.

## Frozen evidence

**results/tables/** contains only aggregate, non-identifying CSV files used by
the manuscript and figures. **results/figures/** contains the final vector
figures. Values in the manuscript should be updated only by rerunning the
analysis and regenerating the corresponding aggregate table.

The prespecified hierarchy is recorded in **docs/ANALYSIS_FREEZE.md**.
Exploratory results must not replace the designated primary comparison.

## Data sources

- NHANES examination data:
  https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
- Public-use linked mortality:
  https://www.cdc.gov/nchs/data-linkage/mortality-public.htm
- Hip accelerometry: NHANES PAXRAW_C and PAXRAW_D
- Later wrist replication: NHANES PAXDAY_G and PAXDAY_H

## Citation and licence

The final article citation and repository DOI should be added after acceptance
or archival release. A project licence must be selected by the authors before
the repository is made public.
