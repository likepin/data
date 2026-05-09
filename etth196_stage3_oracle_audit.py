from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.diagnostics import transform_lambda_splits
from posthoc_calibration.profiles import PROFILES, compute_selected_lambda_splits
from posthoc_calibration.schedules import gamma_from_schedule
from traffic_existing_prediction_ensemble import (
    group_indices,
    group_mean_chunk,
    load_candidates,
    pred_path,
    true_path,
)
from traffic_stage3_lambda_three_source_pilot import (
    build_target_masks,
    compute_dynamic_chunk,
    load_closed_loop_config,
    load_stage2_alpha,
)


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
PROFILE_NAME = "etth196_static_parcorr"
STAGE2_DIR = DATA_ROOT / "deltaA_signal_audit" / "etth196_existing_prediction_ensemble_parcorr"
STAGE2_PREFIX = "etth196_static_parcorr_adaptive_alpha_pilot"
CLOSED_LOOP_DIR = DATA_ROOT / "deltaA_signal_audit" / "etth196_closed_loop_rank_quality_guard_parcorr_ridgebase_sparse"
CLOSED_LOOP_PREFIX = "etth196_static_parcorr_rank_quality_guard_parcorr"
STAGE3_DIRS = {
    "static_p0_dynamic": DATA_ROOT / "deltaA_signal_audit" / "etth196_stage3_lambda_three_source_closed_form_eta2",
    "static_mean_dynamic": DATA_ROOT / "deltaA_signal_audit" / "etth196_stage3_lambda_three_source_closed_form_eta2_staticmean",
}
STAGE3_PREFIXES = {
    "static_p0_dynamic": "etth196_static_parcorr_stage3_closed_form_eta2",
    "static_mean_dynamic": "etth196_static_parcorr_stage3_closed_form_eta2_staticmean",
}
OUT_DIR = DATA_ROOT / "mechanism_evidence" / "etth196_stage3_oracle_audit_20260509"
SEQ_LEN = 96
PRED_LEN = 96
TRAIN_RATIO = 0.7
CHUNK_SIZE = 64


def pct_gain(before: float, after: float) -> float:
    if abs(float(before)) < 1e-12:
        return 0.0
    return 100.0 * (float(before) - float(after)) / float(before)


def read_one(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(frame)}")
    return frame.iloc[0].to_dict()


def fmt_float(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 4) -> str:
    return f"{float(value):+.{digits}f}%"


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "split",
        "variant",
        "oracle",
        "mse",
        "mae",
        "mse_gain_vs_anchor_pct",
        "mae_gain_vs_anchor_pct",
        "active_ratio",
        "active_unit_count",
    ]
    headers = [
        "split",
        "variant",
        "oracle",
        "MSE",
        "MAE",
        "MSE vs anchor",
        "MAE vs anchor",
        "active ratio",
        "active units",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:"]) + " |",
    ]
    for _, row in df[cols].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["split"]),
                    str(row["variant"]),
                    str(row["oracle"]),
                    fmt_float(row["mse"]),
                    fmt_float(row["mae"]),
                    fmt_pct(row["mse_gain_vs_anchor_pct"]),
                    fmt_pct(row["mae_gain_vs_anchor_pct"]),
                    fmt_float(row["active_ratio"], 4),
                    str(int(row["active_unit_count"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def load_gamma_splits(profile: dict) -> dict[str, np.ndarray]:
    lambda_cfg, schedule = load_closed_loop_config(CLOSED_LOOP_DIR, CLOSED_LOOP_PREFIX)
    raw_lambda = compute_selected_lambda_splits(
        profile,
        lambda_cfg=lambda_cfg,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        train_ratio=TRAIN_RATIO,
    )
    lambda_splits = transform_lambda_splits(raw_lambda, lambda_cfg.get("lambda_transform", "raw"))
    return {
        split: gamma_from_schedule(values, schedule).astype(np.float32)
        for split, values in lambda_splits.items()
    }


def selected_stage3_spec(variant: str) -> dict:
    path = STAGE3_DIRS[variant] / f"{STAGE3_PREFIXES[variant]}_test_selected_summary.csv"
    row = read_one(path)
    return {
        "eta_mult": float(row["eta_mult"]),
        "eta_raw": float(row["eta_raw"]),
        "target_mask": str(row["target_mask"]),
        "selected_ensemble": str(row["ensemble"]),
        "eta_clip_reason": str(row["eta_clip_reason"]),
        "dynamic_source": str(row["dynamic_source"]),
    }


def mse_mae(sse: float, sae: float, count: int) -> tuple[float, float]:
    return float(sse) / float(count), float(sae) / float(count)


def row(
    *,
    split: str,
    variant: str,
    oracle: str,
    sse: float,
    sae: float,
    count: int,
    anchor_mse: float,
    anchor_mae: float,
    active_count: int,
    unit_count: int,
    spec: dict,
) -> dict:
    mse, mae = mse_mae(sse, sae, count)
    return {
        "split": split,
        "variant": variant,
        "oracle": oracle,
        "mse": mse,
        "mae": mae,
        "mse_gain_vs_anchor_pct": pct_gain(anchor_mse, mse),
        "mae_gain_vs_anchor_pct": pct_gain(anchor_mae, mae),
        "active_ratio": float(active_count) / float(unit_count) if unit_count else 0.0,
        "active_unit_count": int(active_count),
        "unit_count": int(unit_count),
        "selected_ensemble": spec["selected_ensemble"],
        "eta_mult": spec["eta_mult"],
        "eta_raw": spec["eta_raw"],
        "eta_clip_reason": spec["eta_clip_reason"],
        "target_mask": spec["target_mask"],
        "dynamic_source": spec["dynamic_source"],
    }


def evaluate_variant(
    *,
    split: str,
    variant: str,
    candidates: list[dict],
    alpha: np.ndarray,
    gamma: np.ndarray,
    delta_path: Path,
    target_masks: dict[str, np.ndarray],
) -> list[dict]:
    spec = selected_stage3_spec(variant)
    pred_arrays = [np.load(pred_path(candidate, split), mmap_mode="r") for candidate in candidates]
    true = np.load(true_path(candidates[0], split), mmap_mode="r")
    delta = np.load(delta_path, mmap_mode="r")
    baseline_idx, static_idx = group_indices(candidates)

    n_samples = true.shape[0]
    n_horizon = true.shape[1]
    n_vars = true.shape[2]
    if alpha.size != n_vars:
        raise RuntimeError(f"Alpha length mismatch: {alpha.size} vs {n_vars}")
    if len(gamma) < n_samples:
        raise RuntimeError(f"Gamma too short for {split}: {len(gamma)} vs {n_samples}")
    if delta.shape[0] < n_samples or delta.shape[1:] != (n_vars, n_vars):
        raise RuntimeError(f"Unexpected delta shape for {split}: {delta.shape}")

    mask = target_masks[spec["target_mask"]].reshape(1, 1, -1).astype(np.float32)
    alpha_view = alpha.reshape(1, 1, -1)
    gamma = np.asarray(gamma[:n_samples], dtype=np.float32)
    eta = float(spec["eta_mult"])
    count = int(n_samples * n_horizon * n_vars)

    sums = {
        "anchor": {"sse": 0.0, "sae": 0.0, "active": 0, "units": n_samples},
        "selected_stage3": {"sse": 0.0, "sae": 0.0, "active": n_samples, "units": n_samples},
        "oracle_window_gate": {"sse": 0.0, "sae": 0.0, "active": 0, "units": n_samples},
        "oracle_target_gate": {"sse": 0.0, "sae": 0.0, "active": 0, "units": n_samples * n_vars},
        "oracle_point_gate": {"sse": 0.0, "sae": 0.0, "active": 0, "units": count},
    }

    for start in range(0, n_samples, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_samples)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        anchor = baseline_mean + alpha_view * (static_mean - baseline_mean)
        err_anchor = np.asarray(true[start:end], dtype=np.float32) - anchor
        if spec["dynamic_source"] == "static_mean":
            source_pred = static_mean
        else:
            source_pred = np.asarray(pred_arrays[static_idx[0]][start:end], dtype=np.float32)
        dynamic = compute_dynamic_chunk(source_pred, delta[start:end]) * mask
        corr = eta * gamma[start:end].reshape(-1, 1, 1) * dynamic
        err_stage3 = err_anchor - corr

        abs_anchor = np.abs(err_anchor)
        abs_stage3 = np.abs(err_stage3)
        sq_anchor = np.square(err_anchor, dtype=np.float32)
        sq_stage3 = np.square(err_stage3, dtype=np.float32)

        anchor_sample_sse = sq_anchor.sum(axis=(1, 2), dtype=np.float64)
        stage_sample_sse = sq_stage3.sum(axis=(1, 2), dtype=np.float64)
        anchor_sample_sae = abs_anchor.sum(axis=(1, 2), dtype=np.float64)
        stage_sample_sae = abs_stage3.sum(axis=(1, 2), dtype=np.float64)
        window_active = stage_sample_sse < anchor_sample_sse

        anchor_target_sse = sq_anchor.sum(axis=1, dtype=np.float64)
        stage_target_sse = sq_stage3.sum(axis=1, dtype=np.float64)
        anchor_target_sae = abs_anchor.sum(axis=1, dtype=np.float64)
        stage_target_sae = abs_stage3.sum(axis=1, dtype=np.float64)
        target_active = stage_target_sse < anchor_target_sse

        point_active = sq_stage3 < sq_anchor

        sums["anchor"]["sse"] += float(sq_anchor.sum(dtype=np.float64))
        sums["anchor"]["sae"] += float(abs_anchor.sum(dtype=np.float64))
        sums["selected_stage3"]["sse"] += float(sq_stage3.sum(dtype=np.float64))
        sums["selected_stage3"]["sae"] += float(abs_stage3.sum(dtype=np.float64))

        sums["oracle_window_gate"]["sse"] += float(
            np.where(window_active, stage_sample_sse, anchor_sample_sse).sum(dtype=np.float64)
        )
        sums["oracle_window_gate"]["sae"] += float(
            np.where(window_active, stage_sample_sae, anchor_sample_sae).sum(dtype=np.float64)
        )
        sums["oracle_window_gate"]["active"] += int(window_active.sum())

        sums["oracle_target_gate"]["sse"] += float(
            np.where(target_active, stage_target_sse, anchor_target_sse).sum(dtype=np.float64)
        )
        sums["oracle_target_gate"]["sae"] += float(
            np.where(target_active, stage_target_sae, anchor_target_sae).sum(dtype=np.float64)
        )
        sums["oracle_target_gate"]["active"] += int(target_active.sum())

        sums["oracle_point_gate"]["sse"] += float(
            np.minimum(sq_anchor, sq_stage3).sum(dtype=np.float64)
        )
        sums["oracle_point_gate"]["sae"] += float(
            np.where(point_active, abs_stage3, abs_anchor).sum(dtype=np.float64)
        )
        sums["oracle_point_gate"]["active"] += int(point_active.sum())

    anchor_mse, anchor_mae = mse_mae(sums["anchor"]["sse"], sums["anchor"]["sae"], count)
    return [
        row(
            split=split,
            variant=variant,
            oracle=oracle,
            sse=values["sse"],
            sae=values["sae"],
            count=count,
            anchor_mse=anchor_mse,
            anchor_mae=anchor_mae,
            active_count=values["active"],
            unit_count=values["units"],
            spec=spec,
        )
        for oracle, values in sums.items()
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = dict(PROFILES[PROFILE_NAME])
    interface_dir = Path(profile["interface_dir"])
    alpha = load_stage2_alpha(STAGE2_DIR, STAGE2_PREFIX)
    target_masks = build_target_masks(["all", "top_alpha_14", "top_alpha_29"], alpha)
    candidates = load_candidates(profile)
    gamma_splits = load_gamma_splits(profile)

    rows: list[dict] = []
    for variant in STAGE3_DIRS:
        for split in ("val", "test"):
            rows.extend(
                evaluate_variant(
                    split=split,
                    variant=variant,
                    candidates=candidates,
                    alpha=alpha,
                    gamma=gamma_splits[split],
                    delta_path=interface_dir / f"deltaA_{split}.npy",
                    target_masks=target_masks,
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "etth196_stage3_oracle_audit_table.csv", index=False)
    (OUT_DIR / "etth196_stage3_oracle_audit_table.md").write_text(
        "# ETTh1-96 Stage3 Dynamic Oracle Audit\n\n"
        "This is a diagnostic upper-bound table. Oracle rows use test labels to decide whether the dynamic correction helps, so they are not reportable method performance.\n\n"
        + markdown_table(table)
        + "\n",
        encoding="utf-8",
    )

    test = table[table["split"] == "test"].copy()
    p0_window = test[(test["variant"] == "static_p0_dynamic") & (test["oracle"] == "oracle_window_gate")].iloc[0]
    p0_target = test[(test["variant"] == "static_p0_dynamic") & (test["oracle"] == "oracle_target_gate")].iloc[0]
    p0_point = test[(test["variant"] == "static_p0_dynamic") & (test["oracle"] == "oracle_point_gate")].iloc[0]
    readme_lines = [
        "# ETTh1-96 Stage3 Dynamic Oracle Audit",
        "",
        "Purpose:",
        "- Diagnose whether the current ETTh1 dynamic graph branch contains useful upper-bound signal once the gate is made perfect.",
        "- Keep this separate from method performance: all oracle rows use labels for routing decisions.",
        "",
        "Main test readout using `static_p0_dynamic`:",
        f"- Window-level oracle: `{fmt_float(p0_window['mse'])} / {fmt_float(p0_window['mae'])}`, gain vs adaptive anchor `{fmt_pct(p0_window['mse_gain_vs_anchor_pct'])} / {fmt_pct(p0_window['mae_gain_vs_anchor_pct'])}`, active ratio `{fmt_float(p0_window['active_ratio'], 4)}`.",
        f"- Target-level oracle: `{fmt_float(p0_target['mse'])} / {fmt_float(p0_target['mae'])}`, gain vs adaptive anchor `{fmt_pct(p0_target['mse_gain_vs_anchor_pct'])} / {fmt_pct(p0_target['mae_gain_vs_anchor_pct'])}`, active ratio `{fmt_float(p0_target['active_ratio'], 4)}`.",
        f"- Point-level oracle: `{fmt_float(p0_point['mse'])} / {fmt_float(p0_point['mae'])}`, gain vs adaptive anchor `{fmt_pct(p0_point['mse_gain_vs_anchor_pct'])} / {fmt_pct(p0_point['mae_gain_vs_anchor_pct'])}`, active ratio `{fmt_float(p0_point['active_ratio'], 4)}`.",
        "",
        "Interpretation guide:",
        "- `oracle_window_gate` is the most realistic upper bound for a lambda/window gate.",
        "- `oracle_target_gate` is the upper bound for target-specific `lambda_i` routing.",
        "- `oracle_point_gate` is an aggressive diagnostic ceiling and should not be treated as an implementable route.",
        "",
        "Files:",
        "- `etth196_stage3_oracle_audit_table.csv/md`: full oracle table for `val` and `test`, both dynamic-source variants.",
        "- `manifest.json`: source paths and fixed settings.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "artifact": "etth196_stage3_dynamic_oracle_audit",
                "profile": PROFILE_NAME,
                "stage2_dir": str(STAGE2_DIR),
                "stage2_prefix": STAGE2_PREFIX,
                "closed_loop_dir": str(CLOSED_LOOP_DIR),
                "closed_loop_prefix": CLOSED_LOOP_PREFIX,
                "stage3_dirs": {key: str(value) for key, value in STAGE3_DIRS.items()},
                "output_dir": str(OUT_DIR),
                "seq_len": SEQ_LEN,
                "pred_len": PRED_LEN,
                "train_ratio": TRAIN_RATIO,
                "chunk_size": CHUNK_SIZE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Done] wrote {OUT_DIR}")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
