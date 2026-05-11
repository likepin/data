from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
PACKAGE_DIR = DATA_ROOT / "mechanism_evidence" / "solar96_dynamic_gate_diagnostic_20260511"
TABLE_DIR = PACKAGE_DIR / "tables"
FIG_DIR = PACKAGE_DIR / "figures"
RAW_REF_DIR = PACKAGE_DIR / "raw_refs"

ADEQUACY_DIR = DATA_ROOT / "deltaA_signal_audit" / "solar96_static_lambda_adequacy"
PROBE_DIR = DATA_ROOT / "deltaA_signal_audit" / "solar96_static_lambda_gate_logistic_probe"
ADEQUACY_PREFIX = "solar96_static_lambda_adequacy"
PROBE_PREFIX = "solar96_static_lambda_gate_logistic_probe"


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
    ridge_top5 = one_row(target_topk, model="ridge", split="test", top_frac="0.05")
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
            "interpretation": "Solar lambda is weakly positive on test, but not strong enough to select a deployable active route.",
        },
        {
            "evidence_id": "unit_dynamic_gain_test",
            "split": "test",
            "metric": "raw unit dynamic gain",
            "value": f"mean {fmt(test_split['oracle_unit_mse_gain_mean'])}; positive-rate {fmt_pct(test_split['oracle_unit_positive_rate'])}",
            "interpretation": "The uncalibrated dynamic increment is harmful on average despite a slightly larger positive region than Weather.",
        },
        {
            "evidence_id": "oracle_eta2_gain_test",
            "split": "test",
            "metric": "oracle eta2 gain",
            "value": f"mean {fmt(test_split['oracle_eta2_mse_gain_mean'])}",
            "interpretation": "Solar has clearer recoverable dynamic signal under ideal scaling than Weather.",
        },
        {
            "evidence_id": "selected_gamma_test",
            "split": "test",
            "metric": "selected gamma gain",
            "value": f"mean {fmt(test_split['selected_gamma_mse_gain_mean'])}; active-ratio {fmt_pct(test_split['gamma_active_ratio'])}",
            "interpretation": "The existing closed-loop schedule is safe but very weak on test.",
        },
        {
            "evidence_id": "ridge_gain_regression_test",
            "split": "test",
            "metric": "gain regression generalization",
            "value": f"Pearson {fmt(ridge_test['pearson'])}; Spearman {fmt(ridge_test['spearman'])}; R2 {fmt(ridge_test['r2'])}",
            "interpretation": "Ridge preserves some ranking but its continuous gain calibration does not generalize cleanly.",
        },
        {
            "evidence_id": "huber_gain_regression_test",
            "split": "test",
            "metric": "robust gain regression generalization",
            "value": f"Pearson {fmt(huber_test['pearson'])}; Spearman {fmt(huber_test['spearman'])}; R2 {fmt(huber_test['r2'])}",
            "interpretation": "Huber behaves as a conservative ranker but not a positive gain estimator.",
        },
        {
            "evidence_id": "ridge_top5_test",
            "split": "test",
            "metric": "Ridge top-5% risk-return",
            "value": f"mean {fmt(ridge_top5['oracle_gain_mean_top'])}; positive-rate {fmt_pct(ridge_top5['positive_rate_top'])}; worst5 {fmt(ridge_top5['worst_5pct_gain_mean_top'])}",
            "interpretation": "Top-5% Ridge mostly selects zero-gain/bypass-like rows, not positive dynamic gain.",
        },
        {
            "evidence_id": "ridge_top10_test",
            "split": "test",
            "metric": "Ridge top-10% risk-return",
            "value": f"mean {fmt(ridge_top10['oracle_gain_mean_top'])}; positive-rate {fmt_pct(ridge_top10['positive_rate_top'])}; worst5 {fmt(ridge_top10['worst_5pct_gain_mean_top'])}",
            "interpretation": "Ridge top-10% remains negative, so the active gain frontier is not deployable.",
        },
        {
            "evidence_id": "huber_top10_test",
            "split": "test",
            "metric": "Huber top-10% risk-return",
            "value": f"mean {fmt(huber_top10['oracle_gain_mean_top'])}; positive-rate {fmt_pct(huber_top10['positive_rate_top'])}; worst5 {fmt(huber_top10['worst_5pct_gain_mean_top'])}",
            "interpretation": "Huber avoids the worst loss only by staying near bypass/zero-dynamic behavior.",
        },
        {
            "evidence_id": "ridge_bin_contrast",
            "split": "test",
            "metric": "Ridge top vs bottom decile",
            "value": f"top decile mean {fmt(ridge_bin1['oracle_gain_mean'])}; bottom decile mean {fmt(ridge_bin10['oracle_gain_mean'])}",
            "interpretation": "The risk ranker separates safer windows from catastrophic windows, but top decile is still negative.",
        },
        {
            "evidence_id": "huber_zero_region",
            "split": "test",
            "metric": "Huber top decile",
            "value": f"mean {fmt(huber_bin1['oracle_gain_mean'])}; nonzero-dynamic-rate {fmt_pct(huber_bin1['nonzero_dynamic_rate'])}",
            "interpretation": "The robust route selects a mixed zero/safe region rather than a reliable active gain region.",
        },
        {
            "evidence_id": "val_test_consistency",
            "split": "val/test",
            "metric": "raw dynamic positive-rate",
            "value": f"val {fmt_pct(val_split['oracle_unit_positive_rate'])}; test {fmt_pct(test_split['oracle_unit_positive_rate'])}",
            "interpretation": "The positive raw-dynamic region is sparse but slightly denser than Weather.",
        },
        {
            "evidence_id": "top_coefficients",
            "split": "val-fit",
            "metric": "dominant gain-regression features",
            "value": "; ".join(
                f"{row['model']}:{row['feature']}={fmt(row['coef_standardized'])}" for _, row in top_coef.iterrows()
            ),
            "interpretation": "Dynamic energy/shape dominate gain prediction; static alpha is not the main driver.",
        },
    ]
    return pd.DataFrame(rows)


def build_claim_matrix(evidence: pd.DataFrame) -> pd.DataFrame:
    get = evidence.set_index("evidence_id")
    return pd.DataFrame(
        [
            {
                "candidate_claim": "Solar-96 has stronger dynamic signal than Weather-96.",
                "evidence": get.loc["oracle_eta2_gain_test", "value"] + " / " + get.loc["val_test_consistency", "value"],
                "verdict": "support_with_guard",
                "paper_safe_framing": "Solar shows clearer recoverable dynamic signal, but only under ideal scaling or heavy guard.",
            },
            {
                "candidate_claim": "Solar-96 dynamic branch can be directly promoted to a positive active route.",
                "evidence": get.loc["ridge_top10_test", "value"],
                "verdict": "reject_for_now",
                "paper_safe_framing": "Current deployable gain-aware gates do not produce a positive active frontier.",
            },
            {
                "candidate_claim": "The existing closed-loop schedule is useful but weak.",
                "evidence": get.loc["selected_gamma_test", "value"],
                "verdict": "support",
                "paper_safe_framing": "Closed-loop scheduling contributes a tiny safe correction rather than a strong dynamic route.",
            },
            {
                "candidate_claim": "Risk-return diagnostics justify bypass/guard behavior on Solar.",
                "evidence": get.loc["ridge_bin_contrast", "value"] + " / " + get.loc["huber_zero_region", "value"],
                "verdict": "support",
                "paper_safe_framing": "Gain-aware probes identify safer windows but not reliable positive active corrections.",
            },
            {
                "candidate_claim": "A probability gate alone is enough.",
                "evidence": "Target logistic AUC is useful, but top-k gain/CVaR remains non-positive.",
                "verdict": "reject",
                "paper_safe_framing": "Solar reinforces the need for expected-gain and downside-risk audits.",
            },
            {
                "candidate_claim": "Solar is a better next target than Traffic for refining dynamic gates.",
                "evidence": get.loc["oracle_eta2_gain_test", "value"] + " and tractable 137-variable target-wise diagnostics.",
                "verdict": "support",
                "paper_safe_framing": "Solar is the appropriate medium-scale case for dynamic-gate diagnostics before Traffic-scale deployment.",
            },
        ]
    )


def write_readme(evidence: pd.DataFrame, claims: pd.DataFrame, manifest: dict) -> None:
    lines = [
        "# Solar96 Dynamic Gate Diagnostic Evidence",
        "",
        "Generated: 2026-05-11",
        "",
        "This package freezes the Solar-96 dynamic-gate diagnostic readout using the same lambda adequacy and gain-aware/CVaR probe protocol used for Weather-96.",
        "",
        "## Boundary",
        "",
        "- This is a mechanism and risk diagnostic package, not a new training result.",
        "- Validation is used for fitting lightweight probes; test rows are diagnostic readouts.",
        "- The source `deltaA_signal_audit` directories remain local artifacts and are referenced in `manifest.json`.",
        "",
        "## Executive Summary",
        "",
        "Solar-96 has clearer recoverable dynamic signal than Weather-96 under oracle scaling, but the current deployable lambda/gain gates still do not produce a positive active frontier.",
        "Ridge and Huber can rank safer windows, yet their top selections are zero-gain or negative on average; this supports guarded selective/bypass behavior rather than forced activation.",
        "",
        "Core interpretation:",
        "",
        "> Solar-96 is a medium-scale mixed case: dynamic information is stronger than Weather, but current gate design is still not strong enough for an active performance claim.",
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
        "- `tables/solar96_dynamic_gate_evidence_snapshot.csv`: compact evidence rows.",
        "- `tables/solar96_dynamic_gate_claim_verdict_matrix.csv`: claim-safe interpretation matrix.",
        "- `tables/solar96_static_lambda_adequacy_split_summary.csv`: lambda adequacy split summary.",
        "- `tables/solar96_static_lambda_gate_logistic_probe_target_gain_metrics.csv`: gain regression metrics.",
        "- `tables/solar96_static_lambda_gate_logistic_probe_target_gain_topk_cvar.csv`: top-k and CVaR table.",
        "- `tables/solar96_static_lambda_gate_logistic_probe_target_gain_quantile_bins.csv`: predicted-gain decile bins.",
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
    target_metrics = read_csv(PROBE_DIR / f"{PROBE_PREFIX}_target_gain_metrics.csv")
    target_topk = read_csv(PROBE_DIR / f"{PROBE_PREFIX}_target_gain_topk_cvar.csv")
    target_bins = read_csv(PROBE_DIR / f"{PROBE_PREFIX}_target_gain_quantile_bins.csv")
    target_coef = read_csv(PROBE_DIR / f"{PROBE_PREFIX}_target_gain_coefficients.csv")

    evidence = build_evidence_snapshot(split, target_metrics, target_topk, target_bins, target_coef)
    claims = build_claim_matrix(evidence)

    evidence.to_csv(TABLE_DIR / "solar96_dynamic_gate_evidence_snapshot.csv", index=False)
    claims.to_csv(TABLE_DIR / "solar96_dynamic_gate_claim_verdict_matrix.csv", index=False)
    write_markdown_table(evidence, TABLE_DIR / "solar96_dynamic_gate_evidence_snapshot.md")
    write_markdown_table(claims, TABLE_DIR / "solar96_dynamic_gate_claim_verdict_matrix.md")

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
            "python lambda_adequacy_audit.py --profile solar96_static --tag lambda_adequacy --closed-loop-tag= --adaptive-alpha-csv deltaA_signal_audit\\solar96_existing_prediction_ensemble\\solar96_static_adaptive_alpha_variable_alpha.csv --progress-every 1000",
            "python lambda_gate_logistic_probe.py --profile solar96_static --audit-tag lambda_adequacy --closed-loop-tag= --tag lambda_gate_logistic_probe --adaptive-alpha-csv deltaA_signal_audit\\solar96_existing_prediction_ensemble\\solar96_static_adaptive_alpha_variable_alpha.csv --progress-every 1000",
            "python solar_dynamic_gate_diagnostic_evidence_pack.py",
            "```",
            "",
        ]
    )
    (RAW_REF_DIR / "README.md").write_text(source_readme, encoding="utf-8")

    manifest = {
        "package": "solar96_dynamic_gate_diagnostic_20260511",
        "generated_at": "2026-05-11",
        "dataset": "Solar",
        "horizon": 96,
        "profile": "solar96_static",
        "purpose": "Freeze dynamic-gate mixed/guarded mechanism evidence for Solar-96.",
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
            "solar_dynamic_gate_diagnostic_evidence_pack.py",
        ],
        "key_verdict": "Solar has stronger oracle dynamic signal than Weather, but current deployable gates still justify guarded selective/bypass framing.",
    }
    (PACKAGE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(evidence, claims, manifest)

    print(f"[Done] evidence package written to {PACKAGE_DIR}", flush=True)
    print(evidence.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
