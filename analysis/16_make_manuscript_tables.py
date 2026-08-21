"""Create manuscript-ready tables from generated analysis CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
OUT = HERE / "MANUSCRIPT_TABLES.md"


def fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    if pd.isna(value):
        return ""
    if round(float(value), digits) == 0:
        value = 0.0
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def ci(point: float, low: float, high: float, digits: int = 4, signed: bool = False) -> str:
    return f"{fmt(point, digits, signed)} ({fmt(low, digits, signed)} to {fmt(high, digits, signed)})"


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / f"manuscript_{name}.csv", index=False)


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = []
    for idx, column in enumerate(columns):
        values = [row[idx] for row in rows]
        widths.append(max(len(str(column)), *(len(value) for value in values)))

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    header = fmt_row([str(column) for column in columns])
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [fmt_row(row) for row in rows]
    return "\n".join([header, divider, *body])


def table1() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "cohort_characteristics.csv")
    data = data.rename(columns={"characteristic": "Characteristic", "overall": "Overall"})
    save_table(data, "table1_cohort")
    return data


def table2() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "primary_results.csv")
    contrasts = pd.read_csv(TABLES / "prespecified_comparisons.csv")
    contrast_lookup = {
        (row.model_id, row.reference_id): row
        for row in contrasts.itertuples(index=False)
    }
    rows = []
    for row in data.itertuples(index=False):
        vs_m0 = contrast_lookup.get((row.model_id, "M0"))
        vs_m3 = contrast_lookup.get((row.model_id, "M3"))
        m0_values = (
            (vs_m0.delta_c_index, vs_m0.ci_low, vs_m0.ci_high)
            if vs_m0 is not None
            else (row.delta_vs_M0, row.delta_vs_M0_low, row.delta_vs_M0_high)
        )
        m3_values = (
            (vs_m3.delta_c_index, vs_m3.ci_low, vs_m3.ci_high)
            if vs_m3 is not None
            else (row.delta_vs_M3, row.delta_vs_M3_low, row.delta_vs_M3_high)
        )
        rows.append(
            {
                "ID": row.model_id,
                "Model": row.model_label,
                "C-index (95% CI)": ci(row.c_index, row.ci_low, row.ci_high),
                "Delta C vs PREDICT (95% CI)": ci(
                    *m0_values,
                    signed=True,
                ),
                "Delta C vs circadian (95% CI)": ci(
                    *m3_values,
                    signed=True,
                ),
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "table2_primary_discrimination")
    return out


def table3() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "prespecified_comparisons.csv")
    keep = ["contrast_id", "tier", "question", "model_id", "reference_id", "delta_c_index", "ci_low", "ci_high", "interpretation"]
    data = data[keep]
    rows = []
    for row in data.itertuples(index=False):
        rows.append(
            {
                "Contrast": row.contrast_id,
                "Tier": row.tier,
                "Question": row.question,
                "Model vs reference": f"{row.model_id} vs {row.reference_id}",
                "Delta C (95% CI)": ci(row.delta_c_index, row.ci_low, row.ci_high, signed=True),
                "Interpretation": row.interpretation,
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "table3_prespecified_contrasts")
    return out


def table4() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "paired_primary_clinical_comparison.csv")
    data = data[data["policy"] == "conservative_one_se"].copy()
    metric_label = {
        "c_index": "C-index",
        "calibration_slope_distance": "Calibration slope distance from 1",
        "calibration_in_large_absolute": "Calibration-in-the-large absolute error",
        "brier_10y": "10-year IPCW Brier score",
        "net_benefit_5pct": "Net benefit at 5%",
        "net_benefit_10pct": "Net benefit at 10%",
        "net_benefit_15pct": "Net benefit at 15%",
    }
    rows = []
    for row in data.itertuples(index=False):
        if row.metric == "c_index":
            continue
        digits = 5 if "net_benefit" in row.metric or row.metric == "brier_10y" else 4
        rows.append(
            {
                "Metric": metric_label[row.metric],
                "Direction": "higher is better" if row.direction == "higher_is_better" else "lower is better",
                "M6": fmt(row.M6, digits),
                "M3": fmt(row.M3, digits),
                "Paired difference (95% CI)": ci(row.paired_difference, row.ci_low, row.ci_high, digits, signed=True),
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "table4_primary_clinical_metrics")
    return out


def table5() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "ai_representation_results.csv").set_index("method_id")
    incremental = pd.read_csv(TABLES / "ai_incremental_results.csv").set_index("method_id")
    pairings = [
        ("P8", "CP8", "PCA-8"),
        ("AE8", "CAE8", "Reconstruction AE-8"),
        ("MAE8", "CMAE8", "Masked AE-8"),
        ("CON8", "CCON8", "Contrastive-8"),
        ("RAE8", "CRAE8", "Risk-aware masked AE-8"),
    ]
    rows = []
    for representation_id, combined_id, label in pairings:
        row = data.loc[representation_id]
        combined = incremental.loc[combined_id]
        rows.append(
            {
                "Representation": label,
                "PREDICT + representation C-index": ci(row.c_index, row.ci_low, row.ci_high),
                "Delta C vs fixed circadian B1 (C=0.8444)": ci(
                    row.delta_vs_B1,
                    row.delta_vs_B1_low,
                    row.delta_vs_B1_high,
                    signed=True,
                ),
                "PREDICT + circadian + representation C-index": ci(
                    combined.c_index,
                    combined.ci_low,
                    combined.ci_high,
                ),
                "Added beyond circadian, delta C": ci(
                    combined.delta_vs_B1,
                    combined.delta_vs_B1_low,
                    combined.delta_vs_B1_high,
                    signed=True,
                ),
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "table5_ai_objective_benchmark")
    return out


def table6() -> pd.DataFrame:
    seeds = pd.read_csv(TABLES / "contrastive_seed_stability.csv")
    transport = pd.read_csv(TABLES / "contrastive_cycle_transport.csv")
    rows = [
        {
            "Check": "Seed stability",
            "N": f"{len(seeds)} seeds",
            "C-index": f"{seeds.c_index.mean():.5f} (SD {seeds.c_index.std(ddof=1):.5f})",
            "Delta C vs circadian": f"{seeds.delta_vs_circadian.mean():+.5f} (range {seeds.delta_vs_circadian.min():+.5f} to {seeds.delta_vs_circadian.max():+.5f})",
            "Positive point estimates": f"{int((seeds.delta_vs_circadian > 0).sum())}/{len(seeds)}",
        }
    ]
    for (train, test), group in transport.groupby(["train_cycle", "test_cycle"], sort=True):
        rows.append(
            {
                "Check": f"Train {train}, test {test}",
                "N": f"{int(group.test_participants.iloc[0]):,} participants; {int(group.test_events.iloc[0])} events",
                "C-index": f"{group.c_index.mean():.5f} (range {group.c_index.min():.5f} to {group.c_index.max():.5f})",
                "Delta C vs circadian": f"{group.delta_vs_circadian.mean():+.5f} (range {group.delta_vs_circadian.min():+.5f} to {group.delta_vs_circadian.max():+.5f})",
                "Positive point estimates": f"{int((group.delta_vs_circadian > 0).sum())}/{len(group)}",
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "table6_contrastive_stability_transport")
    return out


def table7() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "ai_representation_readout_r2.csv")
    focus = data[data["method_id"].isin(["AE8", "CON8"])].copy()
    pivot = focus.pivot(index="target", columns="method_id", values="cross_validated_R2").reset_index()
    pivot = pivot.rename(columns={"target": "Known activity feature", "AE8": "Reconstruction AE-8 R2", "CON8": "Contrastive-8 R2"})
    for col in ["Reconstruction AE-8 R2", "Contrastive-8 R2"]:
        pivot[col] = pivot[col].map(lambda value: fmt(value, 2))
    save_table(pivot, "table7_embedding_readout")
    return pivot


def table8() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "horizon_5y_paired_primary_clinical_comparison.csv")
    metric_label = {
        "c_index": "C-index (unchanged from primary analysis)",
        "calibration_slope_distance": "Calibration slope distance from 1",
        "calibration_in_large_absolute": "Calibration-in-the-large absolute error",
        "brier_5y": "5-year IPCW Brier score",
        "net_benefit_5pct": "Net benefit at 5%",
        "net_benefit_10pct": "Net benefit at 10%",
        "net_benefit_15pct": "Net benefit at 15%",
    }
    rows = []
    for row in data.itertuples(index=False):
        digits = 5 if "net_benefit" in row.metric or row.metric == "brier_5y" else 4
        rows.append(
            {
                "Metric": metric_label[row.metric],
                "Direction": "higher is better" if row.direction == "higher_is_better" else "lower is better",
                "M6": fmt(row.M6, digits),
                "M3": fmt(row.M3, digits),
                "Paired difference (95% CI)": ci(
                    row.paired_difference,
                    row.ci_low,
                    row.ci_high,
                    digits,
                    signed=True,
                ),
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "table8_five_year_horizon")
    return out


def table9() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "wrist_cross_device_replication.csv")
    rows = []
    for row in data.itertuples(index=False):
        p_value = "<0.001" if row.p_value < 0.001 else f"{row.p_value:.3f}"
        rows.append(
            {
                "Dataset": row.dataset,
                "Device/activity": row.device,
                "Participants/events": f"{int(row.participants):,} / {int(row.events)}",
                "HR per 1 SD higher activity (95% CI)": ci(
                    row.hr_per_sd, row.hr_ci_low, row.hr_ci_high, digits=2
                ),
                "P value": p_value,
                "PREDICT C-index": fmt(row.predict_c_index, 4),
                "PREDICT + activity C-index": fmt(row.activity_c_index, 4),
                "Delta C (95% CI)": ci(
                    row.delta_c_index,
                    row.delta_ci_low,
                    row.delta_ci_high,
                    digits=4,
                    signed=True,
                ),
                "Role": row.analysis_role,
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "table9_cross_device_replication")
    return out


def table10() -> pd.DataFrame:
    endpoint = pd.read_csv(TABLES / "endpoint_sensitivity.csv").set_index("model_id")
    mapping = pd.read_csv(TABLES / "direct_mapping_sensitivity.csv").set_index("model_id")
    rows = []
    for analysis, source in (
        ("Heart-only mortality endpoint (same 4,341 participants)", endpoint),
        (
            "Direct PREDICT ethnicity mapping (non-Hispanic White in cycles C/D; 2,323 participants; 93 events)",
            mapping,
        ),
    ):
        row = source.loc["M6"]
        rows.append(
            {
                "Sensitivity analysis": analysis,
                "M3 C-index": fmt(source.loc["M3", "c_index"], 4),
                "M6 C-index": fmt(row.c_index, 4),
                "M6 minus M3 (95% CI)": ci(
                    row.delta_vs_M3,
                    row.delta_vs_M3_low,
                    row.delta_vs_M3_high,
                    digits=4,
                    signed=True,
                ),
                "Interpretation": "uncertainty includes no difference",
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "table10_endpoint_mapping_sensitivity")
    return out


def table11() -> pd.DataFrame:
    objective = pd.read_csv(TABLES / "contrastive_objective_comparison.csv").iloc[0]
    ablation = pd.read_csv(TABLES / "contrastive_ablation_summary.csv").set_index("variant")
    rows = [
        {
            "Comparison": objective.comparison,
            "Estimate (95% CI)": ci(
                objective.paired_delta_c,
                objective.ci_low,
                objective.ci_high,
                digits=5,
                signed=True,
            ),
            "Seed direction": "Locked objective comparison",
            "Interpretation": "Suggestive advantage; interval includes zero",
        }
    ]
    interpretations = {
        "no_mask": "No distinct contribution from block masking",
        "no_scale": "No distinct contribution from activity scaling",
        "no_noise": "Noise removal improved performance; post-hoc finding",
        "no_shift": "Time shifting directionally helpful; interval narrowly includes zero",
    }
    for variant in ("no_mask", "no_scale", "no_noise", "no_shift"):
        row = ablation.loc[variant]
        removal = row.label.replace("Remove", "removing", 1).lower()
        rows.append(
            {
                "Comparison": f"C-index loss after {removal}",
                "Estimate (95% CI)": ci(
                    row.mean_c_index_loss_after_removal,
                    row.loss_ci_low,
                    row.loss_ci_high,
                    digits=5,
                    signed=True,
                ),
                "Seed direction": (
                    f"Removal lower in {int(row.seeds_lower_than_full)}/"
                    f"{int(row.seed_count)} seeds"
                ),
                "Interpretation": interpretations[variant],
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "table11_contrastive_mechanism")
    return out


def table12() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "repeated_split_stability.csv")
    comparison_labels = {
        "M6_vs_M3": "Nested AE beyond activity-rhythm features",
        "CCON8_vs_B1": "Fixed contrastive-8 beyond activity-rhythm features",
    }
    rows = []
    for row in data.itertuples(index=False):
        rows.append(
            {
                "Outer-partition seed": int(row.split_seed),
                "Comparison": comparison_labels[row.comparison],
                "Candidate C-index": fmt(row.candidate_c_index, 5),
                "Comparator C-index": fmt(row.comparator_c_index, 5),
                "Delta C (95% CI)": ci(
                    row.delta_c,
                    row.delta_low,
                    row.delta_high,
                    digits=5,
                    signed=True,
                ),
                "Bootstrap P(Delta C > 0)": fmt(
                    row.bootstrap_probability_positive,
                    3,
                ),
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "table12_repeated_split_stability")
    return out


def main() -> None:
    tables = [
        ("Table 2. Cohort characteristics.", table1()),
        ("Table 3. Canonical model ladder and primary discrimination.", table2()),
        ("Table 4. Primary absolute-risk performance and clinical utility, M6 versus M3.", table4()),
        ("Table 5. Exploratory fixed-configuration AI representation-objective benchmark.", table5()),
        ("Table 6. Locked contrastive seed stability and cycle transport.", table6()),
        ("Audit table. Designated paired contrast registry.", table3()),
        ("Supplementary Table S1. Five-year horizon sensitivity, M6 versus M3.", table8()),
        ("Supplementary Table S2. Controlled contrastive-objective and augmentation-ablation analysis.", table11()),
        ("Supplementary Table S3. Endpoint and ethnicity-mapping sensitivities.", table10()),
        ("Supplementary Table S4. Known activity features recoverable from embeddings.", table7()),
        ("Supplementary Table S5. Exploratory cross-device activity-signal replication.", table9()),
        ("Supplementary Table S6. Sensitivity to participant-fold assignment.", table12()),
    ]
    lines = [
        "# Manuscript Tables",
        "",
        "Generated from saved analysis CSV files. Do not manually edit values here; rerun `16_make_manuscript_tables.py` after changing upstream analyses.",
        "The numbering follows the publication-focused structure in `MANUSCRIPT_TABLE_OF_CONTENTS.md`; diagnostic content is retained in the supplement rather than discarded.",
        "",
    ]
    for title, df in tables:
        lines.extend([f"## {title}", "", markdown_table(df), ""])
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")
    for path in sorted(TABLES.glob("manuscript_table*.csv")):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
