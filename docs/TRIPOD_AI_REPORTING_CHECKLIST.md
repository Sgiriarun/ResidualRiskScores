# TRIPOD+AI And PROBAST+AI Working Checklist

This checklist maps the current average-day manuscript package to major
TRIPOD+AI reporting expectations and PROBAST+AI risk-of-bias concerns. It is a
working internal checklist, not a replacement for the official forms.

Status key:

- **Ready**: adequately covered in the current draft or generated tables.
- **Needs detail**: present but should be expanded before submission.
- **Not applicable**: not part of this study design.

## TRIPOD+AI-Oriented Reporting

| Area | Status | Current evidence | Action before submission |
|---|---|---|---|
| Title identifies study type | Ready | Draft title says incremental prognostic value and NHANES. | Add "retrospective" or "nested validation" if journal style prefers. |
| Abstract states objective, data source and model type | Ready | Abstract includes NHANES C/D, PREDICT offset, accelerometry and AI models. | Shorten for journal word limit. |
| Clinical context and intended use | Ready | Drafts describe a future GP workflow as optional risk augmentation after conventional score calculation, and state that current work is preclinical. | Keep this boundary in abstract and discussion. |
| Source data and dates | Ready | Primary NHANES 2003-2006 hip data and supplementary 2011-2014 wrist data use 2019 public linked mortality; files are named in `SUPPLEMENTARY_METHODS.md`. | Keep file names in supplement. |
| Eligibility criteria | Ready | Age 30-79, no prior CVD, required PREDICT variables, valid accelerometry. | Move detailed exclusion counts into Table 1 or supplement. |
| Outcome definition | Ready | Primary endpoint `UCOD_LEADING=001+005`; heart-only sensitivity. | Keep warning that this is a mortality proxy, not full PREDICT endpoint. |
| Predictor definitions | Ready | Volume, MVPA, M10, L5, RA, PCA, AE and contrastive features defined. | Add units consistently in table captions. |
| Missing data handling | Needs detail | Non-wear not zero; cyclic interpolation for never-observed clock minutes; SBP and cholesterol ratio required; missing BMI uses the published category; unavailable or missing binary indicators are encoded as absent. | Add variable-level missingness and exclusion counts before submission. |
| Sample size and events | Ready | 4,341 participants and 192 primary events; cycle and outer-fold event counts are in `SUPPLEMENTARY_METHODS.md`. | Keep fold counts synchronized with generated fold table. |
| Model specification | Ready | Frozen PREDICT offset plus Cox correction from activity features. | Add coefficient tables only for interpretable models if needed. |
| AI model training | Ready | Representation learning inside outer-training folds only. | Add architecture/hyperparameter appendix for reproducibility. |
| Hyperparameter tuning | Ready | Five outer folds, three inner folds, one-SE rule, candidate dimensions and ridge penalties. | Keep maximum-inner-mean sensitivity clearly secondary. |
| Performance metrics | Ready | C-index, calibration slope, calibration-in-the-large, IPCW Brier, decision curves; formulas are summarized in `SUPPLEMENTARY_METHODS.md`. | Convert to journal supplement style if required. |
| Uncertainty intervals | Ready | 1,000 paired participant bootstrap samples; resampling unit is participant. | Add exact seed in final statistical-analysis appendix if journal requires. |
| Calibration and clinical utility | Ready | Calibration/Brier/net benefit reported with claim boundary. | State thresholds are exploratory, not treatment thresholds. |
| Explainability | Ready | Readout R2, correlations and correction phenotypes. | Keep all interpretation descriptive, not causal. |
| Fairness/subgroups | Needs detail | Subgroup table exists. | Report subgroup event counts and avoid fairness claims due to limited events. |
| Reproducibility | Ready | Central OOF prediction files and validation script exist. | Add software versions and computational environment. |
| Data/code availability | Needs detail | Local pipeline exists. | Draft a public-code statement and note NHANES data access route. |

## PROBAST+AI-Oriented Risk-Of-Bias Audit

| Domain | Main concern | Current mitigation | Remaining risk |
|---|---|---|---|
| Participants/data source | NHANES is not the original PREDICT population; the primary model uses two hip cycles. | The paper describes this as transport/model-extension; later wrist cycles are only a directional supplementary check. | The wrist check does not externally validate the learned hip representation or a modern consumer wearable. |
| Outcome | Public NHANES mortality endpoint is not full fatal/non-fatal incident CVD. | Endpoint is labelled as fatal heart-or-cerebrovascular mortality proxy; heart-only sensitivity is separate. | Clinical interpretation must stay conservative. |
| Predictors | PREDICT variables are mapped from NHANES and ethnicity categories differ from New Zealand categories. | Published PREDICT coefficients are frozen; ethnicity mapping sensitivity is included. | Residual mismatch may affect transported LP and absolute-risk calibration. |
| Analysis | Low event count can make AI estimates unstable or optimistic. | Outer-fold training, inner-fold tuning, participant-level bootstrap, seed stability and cycle transport are used. | Contrastive result remains exploratory and needs independent validation. |
| Calibration | Baseline hazard is re-estimated in NHANES for the mortality proxy endpoint. | Fold-specific recalibration avoids using test participants. | This is not validation of original PREDICT absolute risk. |
| Clinical utility | Thresholds are not established treatment cut-points for this mortality endpoint. | Decision curves are explicitly exploratory. | No clinical-deployment claim is supported. |

## Submission-Readiness Gates

Before sending to a journal or senior professor as a full paper, the package
should satisfy these gates:

1. All manuscript numbers match generated CSV/Parquet files.
2. Every result paragraph states whether the analysis is primary, secondary or exploratory.
3. Endpoint language is always "fatal heart-or-cerebrovascular mortality proxy" unless discussing a sensitivity.
4. No text claims incident CVD prediction, GP deployment, or clinical superiority.
5. Figures and tables are generated from saved predictions, not manually entered.
6. The final manuscript cites `REFERENCES.md` sources in the journal's required format.
7. `SUPPLEMENTARY_METHODS.md` includes model architecture summary, hyperparameter grid, fold construction, metric formulas and validation checks.
