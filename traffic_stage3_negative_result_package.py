from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
PACKAGE_ROOT = DATA_ROOT / "mechanism_evidence" / "traffic96_stage3_lambda_three_source_20260507"
OUT_DIR = PACKAGE_ROOT / "interpretation" / "negative_result_explanation"
FROZEN_TABLE = (
    PACKAGE_ROOT
    / "performance"
    / "stage3_lambda_three_source"
    / "tables"
    / "traffic96_static_stage3_lambda_three_source_frozen_table.csv"
)
RISK_GROUP_TABLE = PACKAGE_ROOT / "mechanism" / "risk_windows" / "traffic96_stage3_eta2_risk_group_table.csv"
FOLD_TABLE = PACKAGE_ROOT / "mechanism" / "risk_windows" / "traffic96_stage3_eta2_fold_contribution.csv"
RISK_README = PACKAGE_ROOT / "mechanism" / "risk_windows" / "README.md"


def pct_text(value: float) -> str:
    return f"{float(value):+.4f}%"


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
        raise ValueError(f"Expected one row for {filters}, got {len(rows)}")
    return rows.iloc[0]


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_evidence_snapshot(frozen: pd.DataFrame, risk: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    stage2 = one_row(frozen, selection_role="stage2_anchor")
    grid = one_row(frozen, selection_role="stage3_grid_selected")
    eta2 = one_row(frozen, selection_role="stage3_recommended_closed_form_eta2")
    test_all = one_row(risk, split="test", risk_group="all")
    test_floor = one_row(risk, split="test", risk_group="gamma_floor")
    test_active = one_row(risk, split="test", risk_group="gamma_active_gt_floor")
    test_top5 = one_row(risk, split="test", risk_group="top_rank_5pct_gamma")
    val_active = one_row(risk, split="val", risk_group="gamma_active_gt_floor")
    val_top5 = one_row(risk, split="val", risk_group="top_rank_5pct_gamma")
    val_fold4 = one_row(folds, split="val", fold="4")

    rows = [
        {
            "evidence_id": "performance_stage2_anchor",
            "split": "test",
            "metric": "Stage2 anchor MSE / MAE",
            "value": f"{float(stage2['test_mse']):.10f} / {float(stage2['test_mae']):.10f}",
            "interpretation": "Static/adaptive anchor is the stable Traffic performance base.",
        },
        {
            "evidence_id": "performance_stage3_grid",
            "split": "test",
            "metric": "Stage3 grid gain vs Stage2",
            "value": f"MSE {pct_text(grid['test_mse_gain_vs_stage2_anchor_pct'])}, MAE {pct_text(grid['test_mae_gain_vs_stage2_anchor_pct'])}",
            "interpretation": "Grid Stage3 is weak positive, not a strong new branch.",
        },
        {
            "evidence_id": "performance_stage35_eta2",
            "split": "test",
            "metric": "Stage3.5 closed-form eta2 gain vs Stage2",
            "value": f"MSE {pct_text(eta2['test_mse_gain_vs_stage2_anchor_pct'])}, MAE {pct_text(eta2['test_mae_gain_vs_stage2_anchor_pct'])}",
            "interpretation": "Closed-form eta2 slightly improves grid, but the increment remains tiny.",
        },
        {
            "evidence_id": "risk_all_test",
            "split": "test",
            "metric": "Overall risk-window gain",
            "value": f"MSE {pct_text(test_all['mse_gain_pct'])}",
            "interpretation": "The whole Stage3.5 effect is weak positive.",
        },
        {
            "evidence_id": "risk_gamma_floor_test",
            "split": "test",
            "metric": "gamma_floor coverage / SSE gain share",
            "value": f"{float(test_floor['coverage_pct']):.2f}% / {float(test_floor['sse_gain_share_pct']):.2f}%",
            "interpretation": "Most test gain comes from gamma-floor windows, not active high-risk windows.",
        },
        {
            "evidence_id": "risk_active_test",
            "split": "test",
            "metric": "gamma_active_gt_floor MSE gain",
            "value": pct_text(test_active["mse_gain_pct"]),
            "interpretation": "Active gamma windows are negative on test.",
        },
        {
            "evidence_id": "risk_top5_test",
            "split": "test",
            "metric": "top_rank_5pct_gamma MSE gain",
            "value": pct_text(test_top5["mse_gain_pct"]),
            "interpretation": "The strongest high-gamma windows do not generalize as a positive mechanism.",
        },
        {
            "evidence_id": "risk_active_val",
            "split": "val",
            "metric": "gamma_active_gt_floor MSE gain",
            "value": pct_text(val_active["mse_gain_pct"]),
            "interpretation": "Validation suggests a local active-window opportunity.",
        },
        {
            "evidence_id": "risk_top5_val",
            "split": "val",
            "metric": "top_rank_5pct_gamma MSE gain",
            "value": pct_text(val_top5["mse_gain_pct"]),
            "interpretation": "The validation-side high-gamma signal is real but not test-stable.",
        },
        {
            "evidence_id": "fold4_val",
            "split": "val",
            "metric": "Validation Fold 4 MSE gain",
            "value": pct_text(val_fold4["mse_gain_pct"]),
            "interpretation": "Fold 4 is an anomaly-sensitive validation region, not sufficient test evidence.",
        },
    ]
    return pd.DataFrame(rows)


def build_claim_matrix(frozen: pd.DataFrame, risk: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    eta2 = one_row(frozen, selection_role="stage3_recommended_closed_form_eta2")
    test_floor = one_row(risk, split="test", risk_group="gamma_floor")
    test_active = one_row(risk, split="test", risk_group="gamma_active_gt_floor")
    test_top5 = one_row(risk, split="test", risk_group="top_rank_5pct_gamma")
    val_fold4 = one_row(folds, split="val", fold="4")

    return pd.DataFrame(
        [
            {
                "candidate_claim": "Stage3.5 is a strong Traffic performance module.",
                "evidence": f"Closed-form eta2 test gain vs Stage2 is MSE {pct_text(eta2['test_mse_gain_vs_stage2_anchor_pct'])}, MAE {pct_text(eta2['test_mae_gain_vs_stage2_anchor_pct'])}.",
                "verdict": "reject_strong_claim",
                "paper_safe_framing": "Stage3.5 provides a weak positive add-on over the adaptive static anchor.",
            },
            {
                "candidate_claim": "Lambda-gated dynamics successfully localize high-risk windows on Traffic test.",
                "evidence": f"test gamma_active_gt_floor MSE gain is {pct_text(test_active['mse_gain_pct'])}; top_rank_5pct_gamma MSE gain is {pct_text(test_top5['mse_gain_pct'])}.",
                "verdict": "rejected_on_test",
                "paper_safe_framing": "Current lambda-aware correction does not yet provide reliable high-risk-window localization.",
            },
            {
                "candidate_claim": "The overall gain is driven by active high-gamma windows.",
                "evidence": f"gamma_floor covers {float(test_floor['coverage_pct']):.2f}% of test windows and contributes {float(test_floor['sse_gain_share_pct']):.2f}% of SSE gain.",
                "verdict": "rejected_by_contribution",
                "paper_safe_framing": "Traffic Stage3.5 gain is mostly a weak global / gamma-floor correction effect.",
            },
            {
                "candidate_claim": "Validation Fold 4 evidence is enough to claim test-time risk localization.",
                "evidence": f"Validation Fold 4 MSE gain is {pct_text(val_fold4['mse_gain_pct'])}, but high-gamma active windows are negative on test.",
                "verdict": "not_generalized",
                "paper_safe_framing": "Fold 4 is useful as anomaly evidence, but it does not justify a broad dynamic localization claim.",
            },
            {
                "candidate_claim": "The dynamic branch should become the mainline for Traffic.",
                "evidence": "Stage2 adaptive-alpha remains the stable performance anchor; Stage3.5 adds only a very small post-hoc increment.",
                "verdict": "reject_mainline_shift",
                "paper_safe_framing": "Keep static anchor as the main result; dynamic branch remains guarded and subordinate.",
            },
            {
                "candidate_claim": "The post-hoc guards can be relaxed after Stage3.5.",
                "evidence": "Eta is clipped (`eta_raw=3.670469`, `eta_mult=2.0`) and high-gamma test windows are unstable.",
                "verdict": "reject_guard_relaxation",
                "paper_safe_framing": "Stage3.5 supports guard necessity rather than guard relaxation.",
            },
        ]
    )


def write_report(evidence: pd.DataFrame, claims: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Traffic96 Stage3.5 Negative Result Explanation",
        "",
        "## Executive Summary",
        "",
        "Traffic Stage3.5 is a weak positive result, but a negative mechanism result for the stronger high-risk-window claim.",
        "The closed-form eta2 add-on slightly improves the Stage2 adaptive static anchor, yet Risk Windows show that the test gain is not produced by active high-gamma windows.",
        "",
        "Core conclusion:",
        "",
        "> Stage3.5 should be framed as a small guarded dynamic-aware add-on, not as evidence that lambda-gated dynamics reliably attack high-risk Traffic windows.",
        "",
        "## Key Evidence",
        "",
    ]
    evidence_for_md = evidence[["evidence_id", "split", "metric", "value", "interpretation"]]
    lines.extend(markdown_table_lines(evidence_for_md))
    lines.extend(
        [
            "",
            "## Claim Verdict Matrix",
            "",
        ]
    )
    lines.extend(markdown_table_lines(claims))
    lines.extend(
        [
            "",
            "## Paper-Safe Framing",
            "",
            "Use:",
            "- `Traffic confirms that CACI's guarded post-hoc dynamic branch can produce a small additional correction after a strong adaptive static anchor.`",
            "- `Risk-window diagnostics reveal a boundary condition: the current lambda-gated correction is not reliable in active high-gamma windows on the Traffic test split.`",
            "- `This supports the paper's selective / guarded protocol: dynamic information should be admitted only under validation-selected and guard-constrained conditions.`",
            "",
            "Avoid:",
            "- `Traffic proves the dynamic branch is broadly strong.`",
            "- `Lambda reliably localizes high-risk windows on Traffic test.`",
            "- `High-gamma windows are the main source of the Traffic gain.`",
            "- `Guards can be relaxed after Stage3.5.`",
            "",
            "## Source Tables",
            "",
            f"- Frozen performance table: `{FROZEN_TABLE}`",
            f"- Risk group table: `{RISK_GROUP_TABLE}`",
            f"- Fold contribution table: `{FOLD_TABLE}`",
            f"- Risk Windows README: `{RISK_README}`",
            "",
        ]
    )
    (out_dir / "negative_result_explanation.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table_lines(df: pd.DataFrame) -> list[str]:
    def fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = read_csv(FROZEN_TABLE)
    risk = read_csv(RISK_GROUP_TABLE)
    folds = read_csv(FOLD_TABLE)

    evidence = build_evidence_snapshot(frozen, risk, folds)
    claims = build_claim_matrix(frozen, risk, folds)
    evidence.to_csv(OUT_DIR / "negative_result_evidence_snapshot.csv", index=False)
    claims.to_csv(OUT_DIR / "negative_result_claim_verdict_matrix.csv", index=False)
    write_markdown_table(evidence, OUT_DIR / "negative_result_evidence_snapshot.md")
    write_markdown_table(claims, OUT_DIR / "negative_result_claim_verdict_matrix.md")
    write_report(evidence, claims, OUT_DIR)

    manifest = {
        "package": "traffic96_stage3_negative_result_explanation",
        "date": "2026-05-07",
        "scope": "Traffic96 Stage3.5 closed-form eta2 and Risk Windows interpretation.",
        "status": "negative_mechanism_result_for_high_risk_localization",
        "main_result": "weak_positive_performance_addon",
        "main_negative_result": "active high-gamma windows are not the source of the test gain",
        "source_tables": {
            "frozen_table": str(FROZEN_TABLE),
            "risk_group_table": str(RISK_GROUP_TABLE),
            "fold_table": str(FOLD_TABLE),
            "risk_windows_readme": str(RISK_README),
        },
        "outputs": [
            "negative_result_explanation.md",
            "negative_result_evidence_snapshot.csv",
            "negative_result_evidence_snapshot.md",
            "negative_result_claim_verdict_matrix.csv",
            "negative_result_claim_verdict_matrix.md",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[Wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
