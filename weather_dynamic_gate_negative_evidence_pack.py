from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
PACKAGE_DIR = DATA_ROOT / "mechanism_evidence" / "weather96_dynamic_gate_negative_20260511"
TABLE_DIR = PACKAGE_DIR / "tables"
FIG_DIR = PACKAGE_DIR / "figures"
RAW_REF_DIR = PACKAGE_DIR / "raw_refs"

ADEQUACY_DIR = DATA_ROOT / "deltaA_signal_audit" / "weather96_static_pat3_lambda_adequacy"
PROBE_DIR = DATA_ROOT / "deltaA_signal_audit" / "weather96_static_pat3_lambda_gate_logistic_probe"
ADEQUACY_PREFIX = "weather96_static_pat3_lambda_adequacy"
PROBE_PREFIX = "weather96_static_pat3_lambda_gate_logistic_probe"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def one_row(df: pd.DataFrame, **filters) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, value in filters.items():
        mask &= df[col].astype(str) == str(value)
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row for {filters}, got {len(rows)}")
    return rows.iloc[0]


def fmt(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def markdown_table(df: pd.DataFrame) -> str:
    def cell(value) -> str:
        if isinstance(value, float):
            return f"{value:.6g}" if np.isfinite(value) else "nan"
        return str(value).replace("|", "\\|")

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(markdown_table(df) + "\n", encoding="utf-8")


def git_head(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def copy_file(src: Path, dest: Path) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest.name


def build_evidence_snapshot(
    split: pd.DataFrame,
    target_metrics: pd.DataFrame,
    target_topk: pd.DataFrame,
    target_bins: pd.DataFrame,
    target_coef: pd.DataFrame,
) -> pd.DataFrame:
    val_split = one_row(split, split="val")
    test_split = one_row(split, split="test")
    ridge_test = one_row(target_metrics, model="ridge", split="test")
    huber_test = one_row(target_metrics, model="huber", split="test")
    ridge_top10 = one_row(target_topk, model="ridge", split="test", top_frac="0.1")
    ridge_top1 = one_row(target_topk, model="ridge", split="test", top_frac="0.01")
    huber_top10 = one_row(target_topk, model="huber", split="test", top_frac="0.1")
    ridge_bin1 = one_row(target_bins, model="ridge", split="test", rank_bin="1")
    ridge_bin10 = one_row(target_bins, model="ridge", split="test", rank_bin="10")
    huber_bin1 = one_row(target_bins, model="huber", split="test", rank_bin="1")
    top_coef = target_coef.sort_values(["model", "abs_coef"], ascending=[True, False]).groupby("model").head(2)

    rows = [
        {
            "evidence_id": "lambda_alignment_test",
            "split": "test",
            "metric": "lambda/gamma vs unit oracle MSE gain",
            "value": f"lambda Spearman {fmt(test_split['lambda_spearman_unit_gain'])}; gamma Spearman {fmt(test_split['gamma_spearman_unit_gain'])}",
            "interpretation": "Current lambda is weakly aligned with where raw dynamic correction is truly beneficial.",
        },
        {
            "evidence_id": "unit_dynamic_gain_test",
            "split": "test",
            "metric": "raw unit dynamic gain",
            "value": f"mean {fmt(test_split['oracle_unit_mse_gain_mean'])}; positive-rate {fmt_pct(test_split['oracle_unit_positive_rate'])}",
            "interpretation": "The uncalibrated dynamic increment is mostly harmful on Weather-96.",
        },
        {
            "evidence_id": "oracle_eta2_gain_test",
            "split": "test",
            "metric": "oracle eta2 gain",
            "value": f"mean {fmt(test_split['oracle_eta2_mse_gain_mean'])}",
            "interpretation": "There is weak recoverable signal under ideal scaling, but the magnitude is small.",
        },
        {
            "evidence_id": "ridge_gain_regression_test",
            "split": "test",
            "metric": "gain regression rank quality",
            "value": f"Pearson {fmt(ridge_test['pearson'])}; Spearman {fmt(ridge_test['spearman'])}; R2 {fmt(ridge_test['r2'])}",
            "interpretation": "Ridge learns a strong continuous gain ranking, so the probe is informative.",
        },
        {
            "evidence_id": "huber_gain_regression_test",
            "split": "test",
            "metric": "robust gain regression rank quality",
            "value": f"Pearson {fmt(huber_test['pearson'])}; Spearman {fmt(huber_test['spearman'])}; R2 {fmt(huber_test['r2'])}",
            "interpretation": "Huber learns a conservative risk-avoidance ordering.",
        },
        {
            "evidence_id": "ridge_top10_test",
            "split": "test",
            "metric": "Ridge top-10% risk-return",
            "value": f"mean {fmt(ridge_top10['oracle_gain_mean_top'])}; positive-rate {fmt_pct(ridge_top10['positive_rate_top'])}; worst5 {fmt(ridge_top10['worst_5pct_gain_mean_top'])}",
            "interpretation": "Ridge reduces expected loss sharply but does not turn the selected set positive.",
        },
        {
            "evidence_id": "ridge_top1_test",
            "split": "test",
            "metric": "Ridge top-1% risk-return",
            "value": f"mean {fmt(ridge_top1['oracle_gain_mean_top'])}; positive-rate {fmt_pct(ridge_top1['positive_rate_top'])}; worst5 {fmt(ridge_top1['worst_5pct_gain_mean_top'])}",
            "interpretation": "Even the most optimistic Ridge slice remains negative on average.",
        },
        {
            "evidence_id": "huber_top10_test",
            "split": "test",
            "metric": "Huber top-10% risk-return",
            "value": f"mean {fmt(huber_top10['oracle_gain_mean_top'])}; positive-rate {fmt_pct(huber_top10['positive_rate_top'])}; nonzero dynamics approximately zero in top bin",
            "interpretation": "Huber mainly selects zero-dynamic / zero-gain windows, i.e. safe bypass behavior.",
        },
        {
            "evidence_id": "ridge_bin_contrast",
            "split": "test",
            "metric": "Ridge top vs bottom decile",
            "value": f"top decile mean {fmt(ridge_bin1['oracle_gain_mean'])}; bottom decile mean {fmt(ridge_bin10['oracle_gain_mean'])}",
            "interpretation": "The ranking separates catastrophic negative windows from less harmful windows.",
        },
        {
            "evidence_id": "huber_zero_region",
            "split": "test",
            "metric": "Huber top decile",
            "value": f"mean {fmt(huber_bin1['oracle_gain_mean'])}; nonzero-dynamic-rate {fmt_pct(huber_bin1['nonzero_dynamic_rate'])}",
            "interpretation": "The robust route identifies a near-bypass safety zone rather than an active gain zone.",
        },
        {
            "evidence_id": "val_test_consistency",
            "split": "val/test",
            "metric": "raw dynamic positive-rate",
            "value": f"val {fmt_pct(val_split['oracle_unit_positive_rate'])}; test {fmt_pct(test_split['oracle_unit_positive_rate'])}",
            "interpretation": "The positive raw-dynamic region is sparse on both splits.",
        },
        {
            "evidence_id": "top_coefficients",
            "split": "val-fit",
            "metric": "dominant gain-regression features",
            "value": "; ".join(
                f"{row['model']}:{row['feature']}={fmt(row['coef_standardized'])}" for _, row in top_coef.iterrows()
            ),
            "interpretation": "Dynamic energy/shape dominate gain prediction; lambda_rank is not the main signal.",
        },
    ]
    return pd.DataFrame(rows)


def build_claim_matrix(evidence: pd.DataFrame) -> pd.DataFrame:
    get = evidence.set_index("evidence_id")
    return pd.DataFrame(
        [
            {
                "candidate_claim": "Weather-96 dynamic branch can be made a stable positive performance route by better lambda gating.",
                "evidence": get.loc["ridge_top10_test", "value"],
                "verdict": "reject_for_now",
                "paper_safe_framing": "Dynamic correction is diagnosable but should remain guard-suppressed on Weather-96.",
            },
            {
                "candidate_claim": "The current lambda_rank is the right primary gate signal.",
                "evidence": get.loc["lambda_alignment_test", "value"],
                "verdict": "reject",
                "paper_safe_framing": "lambda_rank is a weak risk proxy; dynamic energy/shape carries more diagnostic information.",
            },
            {
                "candidate_claim": "A probability gate is sufficient.",
                "evidence": "Logistic probe improves positive-rate, but gain-aware top-k remains negative on average.",
                "verdict": "reject",
                "paper_safe_framing": "Expected gain and downside risk must be audited, not only hit probability.",
            },
            {
                "candidate_claim": "Gain-aware regression proves a deployable positive dynamic route.",
                "evidence": get.loc["ridge_gain_regression_test", "value"] + " / " + get.loc["ridge_top1_test", "value"],
                "verdict": "not_supported",
                "paper_safe_framing": "Gain regression is useful for risk ordering, but not sufficient for positive Weather-96 deployment.",
            },
            {
                "candidate_claim": "Huber can rescue the dynamic branch.",
                "evidence": get.loc["huber_top10_test", "value"],
                "verdict": "reject_as_gain_route",
                "paper_safe_framing": "Huber behaves as a conservative bypass selector.",
            },
            {
                "candidate_claim": "Weather-96 should be used as a negative mechanism case.",
                "evidence": get.loc["unit_dynamic_gain_test", "value"] + " / " + get.loc["oracle_eta2_gain_test", "value"],
                "verdict": "support",
                "paper_safe_framing": "Weather-96 supports the guard philosophy: dynamic information exists, but forced activation is unsafe.",
            },
        ]
    )


def write_readme(evidence: pd.DataFrame, claims: pd.DataFrame, manifest: dict) -> None:
    lines = [
        "# Weather96 Dynamic Gate Negative Evidence",
        "",
        "Generated: 2026-05-11",
        "",
        "This package freezes the Weather-96 diagnostic evidence for why the current lambda/dynamic branch should not be promoted to a standalone performance route.",
        "It complements the Weather MSE-primary target-gate performance package by isolating the dynamic-gate failure mode.",
        "",
        "## Boundary",
        "",
        "- This is a mechanism and risk diagnostic package, not a new forecasting result.",
        "- Validation is used for fitting the lightweight probes; test rows are diagnostic readouts.",
        "- The large `deltaA_signal_audit` source directories remain local artifacts and are referenced in `manifest.json`.",
        "",
        "## Executive Summary",
        "",
        "The Weather-96 dynamic branch contains weak recoverable signal under ideal scaling, but the uncalibrated dynamic increment is mostly harmful.",
        "Logistic and gain-aware probes can rank risk and reduce expected damage, yet Ridge top-k selections remain negative on average and Huber mostly chooses zero-dynamic bypass regions.",
        "",
        "Core interpretation:",
        "",
        "> Weather-96 supports the CACI guard philosophy: dynamic information is detectable, but forced dynamic activation is unsafe; static/adaptive anchor should remain primary.",
        "",
        "## Evidence Snapshot",
        "",
        markdown_table(evidence),
        "",
        "## Claim Verdict Matrix",
        "",
        markdown_table(claims),
        "",
        "## Key Files",
        "",
        "- `tables/weather96_dynamic_gate_evidence_snapshot.csv`: compact evidence rows.",
        "- `tables/weather96_dynamic_gate_claim_verdict_matrix.csv`: claim-safe interpretation matrix.",
        "- `tables/weather96_static_pat3_lambda_adequacy_split_summary.csv`: lambda adequacy split summary.",
        "- `tables/weather96_static_pat3_lambda_gate_logistic_probe_target_gain_metrics.csv`: gain regression metrics.",
        "- `tables/weather96_static_pat3_lambda_gate_logistic_probe_target_gain_topk_cvar.csv`: top-k and CVaR table.",
        "- `tables/weather96_static_pat3_lambda_gate_logistic_probe_target_gain_quantile_bins.csv`: predicted-gain decile bins.",
        "- `figures/*risk_return_frontier.png`: risk-return frontier plots.",
        "- `figures/*top5_gain_distribution.png`: selected top-5% gain distribution diagnostics.",
        "",
        "## Source Pointers",
        "",
        f"- adequacy source: `{manifest['source_dirs']['lambda_adequacy']}`",
        f"- probe source: `{manifest['source_dirs']['lambda_gate_probe']}`",
        f"- git head: `{manifest['git_head']}`",
        "",
    ]
    (PACKAGE_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for path in (TABLE_DIR, FIG_DIR, RAW_REF_DIR):
        path.mkdir(parents=True, exist_ok=True)

    split = read_csv(ADEQUACY_DIR / f"{ADEQUACY_PREFIX}_split_summary.csv")
    align = read_csv(ADEQUACY_DIR / f"{ADEQUACY_PREFIX}_feature_alignment.csv")
    target_metrics = read_csv(PROBE_DIR / f"{PROBE_PREFIX}_target_gain_metrics.csv")
    target_topk = read_csv(PROBE_DIR / f"{PROBE_PREFIX}_target_gain_topk_cvar.csv")
    target_bins = read_csv(PROBE_DIR / f"{PROBE_PREFIX}_target_gain_quantile_bins.csv")
    target_coef = read_csv(PROBE_DIR / f"{PROBE_PREFIX}_target_gain_coefficients.csv")
    target_logistic = read_csv(PROBE_DIR / f"{PROBE_PREFIX}_target_metrics.csv")

    evidence = build_evidence_snapshot(split, target_metrics, target_topk, target_bins, target_coef)
    claims = build_claim_matrix(evidence)

    evidence.to_csv(TABLE_DIR / "weather96_dynamic_gate_evidence_snapshot.csv", index=False)
    claims.to_csv(TABLE_DIR / "weather96_dynamic_gate_claim_verdict_matrix.csv", index=False)
    write_markdown_table(evidence, TABLE_DIR / "weather96_dynamic_gate_evidence_snapshot.md")
    write_markdown_table(claims, TABLE_DIR / "weather96_dynamic_gate_claim_verdict_matrix.md")

    table_sources = [
        ADEQUACY_DIR / f"{ADEQUACY_PREFIX}_split_summary.csv",
        ADEQUACY_DIR / f"{ADEQUACY_PREFIX}_feature_alignment.csv",
        PROBE_DIR / f"{PROBE_PREFIX}_target_metrics.csv",
        PROBE_DIR / f"{PROBE_PREFIX}_target_coefficients.csv",
        PROBE_DIR / f"{PROBE_PREFIX}_target_gain_metrics.csv",
        PROBE_DIR / f"{PROBE_PREFIX}_target_gain_coefficients.csv",
        PROBE_DIR / f"{PROBE_PREFIX}_target_gain_topk_cvar.csv",
        PROBE_DIR / f"{PROBE_PREFIX}_target_gain_quantile_bins.csv",
        PROBE_DIR / f"{PROBE_PREFIX}_window_gain_topk_cvar.csv",
        PROBE_DIR / f"{PROBE_PREFIX}_window_gain_quantile_bins.csv",
    ]
    copied_tables = [copy_file(src, TABLE_DIR / src.name) for src in table_sources]

    figure_sources = [
        PROBE_DIR / f"{PROBE_PREFIX}_target_gain_risk_return_frontier.png",
        PROBE_DIR / f"{PROBE_PREFIX}_target_test_top5_gain_distribution.png",
        PROBE_DIR / f"{PROBE_PREFIX}_window_gain_risk_return_frontier.png",
        PROBE_DIR / f"{PROBE_PREFIX}_window_test_top5_gain_distribution.png",
        ADEQUACY_DIR / f"{ADEQUACY_PREFIX}_feature_alignment_spearman.png",
    ]
    copied_figures = [copy_file(src, FIG_DIR / src.name) for src in figure_sources]

    source_readme = "\n".join(
        [
            "# Raw Source References",
            "",
            "Large diagnostic source directories are intentionally not copied in full.",
            "",
            f"- lambda adequacy: `{ADEQUACY_DIR}`",
            f"- lambda gate probe: `{PROBE_DIR}`",
            "",
            "Rebuild commands:",
            "",
            "```powershell",
            "python lambda_adequacy_audit.py --profile weather96_static_pat3 --tag lambda_adequacy --closed-loop-tag full_guard_v2 --progress-every 1000",
            "python lambda_gate_logistic_probe.py --profile weather96_static_pat3 --audit-tag lambda_adequacy --closed-loop-tag full_guard_v2 --tag lambda_gate_logistic_probe --progress-every 1000",
            "python weather_dynamic_gate_negative_evidence_pack.py",
            "```",
            "",
        ]
    )
    (RAW_REF_DIR / "README.md").write_text(source_readme, encoding="utf-8")

    manifest = {
        "package": "weather96_dynamic_gate_negative_20260511",
        "generated_at": "2026-05-11",
        "dataset": "Weather",
        "horizon": 96,
        "profile": "weather96_static_pat3",
        "purpose": "Freeze dynamic-gate negative mechanism evidence for Weather-96.",
        "git_head": git_head(DATA_ROOT),
        "source_dirs": {
            "lambda_adequacy": str(ADEQUACY_DIR),
            "lambda_gate_probe": str(PROBE_DIR),
        },
        "copied_tables": copied_tables,
        "copied_figures": copied_figures,
        "source_scripts": [
            "lambda_adequacy_audit.py",
            "lambda_gate_logistic_probe.py",
            "weather_dynamic_gate_negative_evidence_pack.py",
        ],
        "key_verdict": "Dynamic information is detectable but unsafe to force on Weather-96; use guarded bypass/static-adaptive anchor framing.",
    }
    (PACKAGE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(evidence, claims, manifest)

    print(f"[Done] evidence package written to {PACKAGE_DIR}", flush=True)
    print(evidence.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
