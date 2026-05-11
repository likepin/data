from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.profiles import (
    PROFILES,
    RESULT_ROOT,
    compute_selected_lambda_splits,
    dynamic_args,
)
from posthoc_calibration.schedules import gamma_from_schedule
from posthoc_ecl96_deltaA_manual_gate import (
    build_dynamic_cache,
    load_ecl_zscore,
    resolve_split_ranges,
)


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether lambda/gamma and available risk features align with oracle dynamic gain."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="weather96_static_pat3")
    parser.add_argument("--tag", default="lambda_adequacy")
    parser.add_argument("--closed-loop-tag", default="full_guard_v2")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--eta-max", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--top-fracs", default="0.01,0.05,0.10,0.20")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--adaptive-alpha-csv", default="")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def read_one_csv(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(frame)}")
    return frame.iloc[0].to_dict()


def run_prefix(profile: str, tag: str) -> str:
    return f"{profile}_{tag}" if tag else profile


def default_alpha_csv(profile_name: str) -> Path | None:
    if profile_name == "weather96_static_pat3":
        return (
            DATA_ROOT
            / "deltaA_signal_audit"
            / "weather96_pat3_existing_prediction_ensemble"
            / "weather96_static_pat3_adaptive_alpha_variable_alpha.csv"
        )
    return None


def load_input_shift_features(
    profile: dict,
    split: str,
    n_samples: int,
    seq_len: int,
) -> pd.DataFrame:
    interface_dir = Path(profile["interface_dir"])
    manifest = json.loads((interface_dir / "interface_manifest.json").read_text(encoding="utf-8"))
    geom = manifest["window_geometry"]
    dataset_contract = manifest.get("dataset_contract", {})
    columns = dataset_contract["columns"]
    split_ranges = resolve_split_ranges(manifest, seq_len=seq_len)
    eval_border1 = int(split_ranges[split]["border1"])
    train_end = int(geom["train_interval"][1])
    full_z = load_ecl_zscore(
        Path(profile["data_csv"]),
        columns=columns,
        train_end=train_end,
        date_col=dataset_contract.get("date_col", "date"),
        header_mode=dataset_contract.get("header_mode", "infer"),
        sep=dataset_contract.get("sep", ","),
    )

    abs_mean = np.zeros(n_samples, dtype=np.float64)
    sq_mean = np.zeros(n_samples, dtype=np.float64)
    delta_abs_mean = np.zeros(n_samples, dtype=np.float64)
    for sample_id in range(n_samples):
        window = full_z[eval_border1 + sample_id : eval_border1 + sample_id + seq_len]
        abs_mean[sample_id] = float(np.mean(np.abs(window)))
        sq_mean[sample_id] = float(np.mean(np.square(window)))
        if len(window) > 1:
            delta_abs_mean[sample_id] = float(np.mean(np.abs(np.diff(window, axis=0))))
    return pd.DataFrame(
        {
            "input_abs_mean": abs_mean,
            "input_sq_mean": sq_mean,
            "input_delta_abs_mean": delta_abs_mean,
        }
    )


def audit_split(
    profile_name: str,
    profile: dict,
    split: str,
    lambda_values: np.ndarray,
    schedule: dict,
    lambda_cfg: dict,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[Stage] build dynamic cache split={split}", flush=True)
    dynamic, _legacy_lambda, _schedule_lambda, static_dirs = build_dynamic_cache(
        dynamic_args(profile, split=split, pred_len=args.pred_len, progress_every=args.progress_every)
    )
    n_samples, pred_len, n_vars = dynamic.shape
    if len(lambda_values) != n_samples:
        raise RuntimeError(f"Lambda length mismatch for {split}: {len(lambda_values)} vs {n_samples}")

    gamma = gamma_from_schedule(lambda_values, schedule).astype(np.float64)
    gamma_floor = float(schedule["gamma_min"])
    dyn = np.asarray(dynamic, dtype=np.float32)
    dyn_sq = np.square(dyn, dtype=np.float32)
    dyn_sq_sample_sum = dyn_sq.sum(axis=(1, 2), dtype=np.float64)
    dyn_sq_target_sum = dyn_sq.sum(axis=(0, 1), dtype=np.float64)
    dynamic_energy = dyn_sq_sample_sum / float(pred_len * n_vars)
    dynamic_abs_mean = np.abs(dyn).sum(axis=(1, 2), dtype=np.float64) / float(pred_len * n_vars)

    n_proj = len(static_dirs)
    count_sample = float(n_proj * pred_len * n_vars)
    count_target = float(n_proj * n_samples * pred_len)
    sample_static_sse = np.zeros(n_samples, dtype=np.float64)
    sample_static_sae = np.zeros(n_samples, dtype=np.float64)
    sample_align = np.zeros(n_samples, dtype=np.float64)
    sample_abs_gain_unit = np.zeros(n_samples, dtype=np.float64)
    target_static_sse = np.zeros(n_vars, dtype=np.float64)
    target_align = np.zeros(n_vars, dtype=np.float64)
    target_abs_gain_unit = np.zeros(n_vars, dtype=np.float64)

    pred_file = "val_pred.npy" if split == "val" else "pred.npy"
    true_file = "val_true.npy" if split == "val" else "true.npy"
    expected_shape = (n_samples, pred_len, n_vars)
    for projection, directory in enumerate(static_dirs):
        pred = np.load(Path(directory) / pred_file, mmap_mode="r")
        true = np.load(Path(directory) / true_file, mmap_mode="r")
        if pred.shape != expected_shape:
            raise RuntimeError(f"Unexpected pred shape in {directory}: {pred.shape}, expected {expected_shape}")
        err = np.asarray(true, dtype=np.float32) - np.asarray(pred, dtype=np.float32)
        err_sq = np.square(err, dtype=np.float32)
        align = err * dyn
        abs_gain_unit = np.abs(err).astype(np.float32) - np.abs(err - dyn).astype(np.float32)
        sample_static_sse += err_sq.sum(axis=(1, 2), dtype=np.float64)
        sample_static_sae += np.abs(err).sum(axis=(1, 2), dtype=np.float64)
        sample_align += align.sum(axis=(1, 2), dtype=np.float64)
        sample_abs_gain_unit += abs_gain_unit.sum(axis=(1, 2), dtype=np.float64)
        target_static_sse += err_sq.sum(axis=(0, 1), dtype=np.float64)
        target_align += align.sum(axis=(0, 1), dtype=np.float64)
        target_abs_gain_unit += abs_gain_unit.sum(axis=(0, 1), dtype=np.float64)
        del err, err_sq, align, abs_gain_unit
        print(f"[Split:{split}] projection {projection + 1}/{n_proj}", flush=True)

    dyn_den_sample = np.maximum(float(n_proj) * dyn_sq_sample_sum, 1e-12)
    eta_raw = sample_align / dyn_den_sample
    eta_clipped = np.clip(eta_raw, 0.0, float(args.eta_max))
    unit_gain_sum = 2.0 * sample_align - float(n_proj) * dyn_sq_sample_sum
    selected_gamma_gain_sum = (
        2.0 * gamma * sample_align - np.square(gamma) * float(n_proj) * dyn_sq_sample_sum
    )
    eta_gain_sum = (
        2.0 * eta_clipped * sample_align
        - np.square(eta_clipped) * float(n_proj) * dyn_sq_sample_sum
    )

    sample_df = pd.DataFrame(
        {
            "profile": profile_name,
            "split": split,
            "sample_id": np.arange(n_samples, dtype=np.int64),
            "lambda_value": lambda_values.astype(np.float64),
            "lambda_rank": pd.Series(lambda_values).rank(pct=True, method="average").to_numpy(dtype=np.float64),
            "gamma_selected": gamma,
            "gamma_active": gamma > (gamma_floor + 1e-6),
            "dynamic_energy": dynamic_energy,
            "dynamic_abs_mean": dynamic_abs_mean,
            "static_mse_oracle": sample_static_sse / count_sample,
            "static_mae_oracle": sample_static_sae / count_sample,
            "oracle_unit_mse_gain": unit_gain_sum / count_sample,
            "oracle_unit_mae_gain": sample_abs_gain_unit / count_sample,
            "oracle_eta_raw": eta_raw,
            "oracle_eta_clipped": eta_clipped,
            "oracle_eta2_mse_gain": eta_gain_sum / count_sample,
            "selected_gamma_mse_gain": selected_gamma_gain_sum / count_sample,
            "alignment_mean": sample_align / count_sample,
        }
    )
    sample_df = pd.concat(
        [sample_df, load_input_shift_features(profile, split=split, n_samples=n_samples, seq_len=args.seq_len)],
        axis=1,
    )

    target_df = pd.DataFrame(
        {
            "profile": profile_name,
            "split": split,
            "target_index": np.arange(n_vars, dtype=np.int64),
            "dynamic_energy": dyn_sq_target_sum / float(n_samples * pred_len),
            "static_mse_oracle": target_static_sse / count_target,
            "oracle_unit_mse_gain": (2.0 * target_align - float(n_proj) * dyn_sq_target_sum) / count_target,
            "oracle_unit_mae_gain": target_abs_gain_unit / count_target,
            "alignment_mean": target_align / count_target,
        }
    )
    return sample_df, target_df


def feature_alignment(sample_df: pd.DataFrame) -> pd.DataFrame:
    features = [
        ("lambda_value", True),
        ("lambda_rank", True),
        ("gamma_selected", True),
        ("gamma_active", True),
        ("dynamic_energy", True),
        ("dynamic_abs_mean", True),
        ("input_abs_mean", True),
        ("input_sq_mean", True),
        ("input_delta_abs_mean", True),
        ("static_mse_oracle", False),
        ("static_mae_oracle", False),
    ]
    outcomes = ["oracle_unit_mse_gain", "oracle_eta2_mse_gain", "selected_gamma_mse_gain"]
    rows = []
    for (profile, split), group in sample_df.groupby(["profile", "split"], sort=False):
        for feature, known_at_inference in features:
            for outcome in outcomes:
                corr = group[[feature, outcome]].corr(method="spearman").iloc[0, 1]
                rows.append(
                    {
                        "profile": profile,
                        "split": split,
                        "feature": feature,
                        "outcome": outcome,
                        "spearman": float(corr) if pd.notna(corr) else np.nan,
                        "known_at_inference": bool(known_at_inference),
                    }
                )
    return pd.DataFrame(rows)


def topk_lift(sample_df: pd.DataFrame, top_fracs: list[float]) -> pd.DataFrame:
    features = [
        "lambda_rank",
        "gamma_selected",
        "dynamic_energy",
        "dynamic_abs_mean",
        "input_sq_mean",
        "input_delta_abs_mean",
        "static_mse_oracle",
    ]
    outcomes = ["oracle_unit_mse_gain", "oracle_eta2_mse_gain", "selected_gamma_mse_gain"]
    rows = []
    for (profile, split), group in sample_df.groupby(["profile", "split"], sort=False):
        n = len(group)
        for feature in features:
            order = group[feature].to_numpy(dtype=np.float64).argsort(kind="mergesort")[::-1]
            for frac in top_fracs:
                k = max(1, int(round(frac * n)))
                idx = order[:k]
                for outcome in outcomes:
                    values = group[outcome].to_numpy(dtype=np.float64)
                    positive_values = np.maximum(values, 0.0)
                    positive_total = float(positive_values.sum())
                    top_mean = float(values[idx].mean())
                    all_mean = float(values.mean())
                    rows.append(
                        {
                            "profile": profile,
                            "split": split,
                            "feature": feature,
                            "top_frac": float(frac),
                            "top_n": int(k),
                            "outcome": outcome,
                            "top_mean": top_mean,
                            "all_mean": all_mean,
                            "lift_vs_all": top_mean / all_mean if abs(all_mean) > 1e-12 else np.nan,
                            "positive_rate_top": float(np.mean(values[idx] > 0.0)),
                            "positive_rate_all": float(np.mean(values > 0.0)),
                            "positive_gain_capture_share": (
                                float(positive_values[idx].sum() / positive_total)
                                if positive_total > 1e-12
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def split_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (profile, split), group in sample_df.groupby(["profile", "split"], sort=False):
        rows.append(
            {
                "profile": profile,
                "split": split,
                "n": int(len(group)),
                "lambda_spearman_unit_gain": float(
                    group[["lambda_rank", "oracle_unit_mse_gain"]].corr(method="spearman").iloc[0, 1]
                ),
                "gamma_spearman_unit_gain": float(
                    group[["gamma_selected", "oracle_unit_mse_gain"]].corr(method="spearman").iloc[0, 1]
                ),
                "dynamic_energy_spearman_unit_gain": float(
                    group[["dynamic_energy", "oracle_unit_mse_gain"]].corr(method="spearman").iloc[0, 1]
                ),
                "input_shift_spearman_unit_gain": float(
                    group[["input_sq_mean", "oracle_unit_mse_gain"]].corr(method="spearman").iloc[0, 1]
                ),
                "static_risk_spearman_unit_gain": float(
                    group[["static_mse_oracle", "oracle_unit_mse_gain"]].corr(method="spearman").iloc[0, 1]
                ),
                "oracle_unit_mse_gain_mean": float(group["oracle_unit_mse_gain"].mean()),
                "oracle_eta2_mse_gain_mean": float(group["oracle_eta2_mse_gain"].mean()),
                "selected_gamma_mse_gain_mean": float(group["selected_gamma_mse_gain"].mean()),
                "oracle_unit_positive_rate": float(np.mean(group["oracle_unit_mse_gain"] > 0.0)),
                "gamma_active_ratio": float(np.mean(group["gamma_active"])),
                "gamma_active_unit_gain_mean": float(
                    group.loc[group["gamma_active"], "oracle_unit_mse_gain"].mean()
                )
                if bool(group["gamma_active"].any())
                else np.nan,
                "gamma_inactive_unit_gain_mean": float(
                    group.loc[~group["gamma_active"], "oracle_unit_mse_gain"].mean()
                )
                if bool((~group["gamma_active"]).any())
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_alpha_to_target_summary(target_df: pd.DataFrame, alpha_csv: Path | None) -> pd.DataFrame:
    if alpha_csv is None or not alpha_csv.exists():
        return target_df
    alpha = pd.read_csv(alpha_csv)
    keep = [col for col in ["target_index", "alpha_raw", "alpha_clipped", "alpha_shrunk", "reliability"] if col in alpha]
    if "target_index" not in keep:
        return target_df
    return target_df.merge(alpha[keep], on="target_index", how="left")


def write_figures(out_dir: Path, prefix: str, sample_df: pd.DataFrame, align_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[Warn] skip figures: {exc}", flush=True)
        return

    for split, group in sample_df.groupby("split", sort=False):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(group["lambda_rank"], group["oracle_unit_mse_gain"], s=4, alpha=0.35)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("lambda rank")
        ax.set_ylabel("unit dynamic oracle MSE gain")
        ax.set_title(f"{prefix} {split}: lambda vs oracle gain")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{split}_lambda_vs_oracle_gain.png", dpi=160)
        plt.close(fig)

    pivot = align_df[align_df["outcome"] == "oracle_unit_mse_gain"].pivot(
        index="feature", columns="split", values="spearman"
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman vs unit oracle gain")
    ax.set_title(f"{prefix}: feature alignment")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_feature_alignment_spearman.png", dpi=160)
    plt.close(fig)


def write_readme(out_dir: Path, prefix: str, summary: pd.DataFrame, align: pd.DataFrame, topk: pd.DataFrame) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_empty_"
        cols = list(frame.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in frame.iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.6g}" if np.isfinite(value) else "nan")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        f"# {prefix} Lambda Adequacy Audit",
        "",
        "This diagnostic checks whether the existing lambda/gamma and other lightweight risk features align with windows where the dynamic increment would have oracle-positive MSE effect.",
        "",
        "Important boundary: this is an audit, not a valid test-selection protocol. Test rows are diagnostic only.",
        "",
        "## Split Summary",
        "",
        markdown_table(summary),
        "",
        "## Best Known-at-Inference Feature Alignment",
        "",
    ]
    known = align[(align["known_at_inference"]) & (align["outcome"] == "oracle_unit_mse_gain")].copy()
    known["abs_spearman"] = known["spearman"].abs()
    best = known.sort_values(["split", "abs_spearman"], ascending=[True, False]).groupby("split").head(5)
    lines.append(markdown_table(best.drop(columns=["abs_spearman"])))
    lines.extend(
        [
            "",
            "## Top-10% Lift Against Unit Oracle Gain",
            "",
        ]
    )
    top10 = topk[(topk["top_frac"] == 0.10) & (topk["outcome"] == "oracle_unit_mse_gain")].copy()
    lines.append(markdown_table(top10))
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- If `lambda_rank` / `gamma_selected` align weakly with oracle gain but dynamic-energy or input-shift features align better, current lambda is likely under-specified.",
            "- If all known-at-inference features align weakly while oracle eta gain is still tiny, the dynamic asset itself is likely too thin for a performance mainline.",
            "- `static_mse_oracle` is not known at inference; it is included only as an upper diagnostic reference.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    out_dir = Path(args.out_dir) if args.out_dir else DATA_ROOT / "deltaA_signal_audit" / f"{args.profile}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_prefix(args.profile, args.tag)
    closed_loop_prefix = run_prefix(args.profile, args.closed_loop_tag)
    closed_loop_dir = Path(profile["out_dir"])
    lambda_cfg = read_one_csv(closed_loop_dir / f"{closed_loop_prefix}_closed_loop_lambda_selected.csv")
    schedule = read_one_csv(closed_loop_dir / f"{closed_loop_prefix}_closed_loop_schedule_selected.csv")
    lambda_splits = compute_selected_lambda_splits(
        profile,
        lambda_cfg=lambda_cfg,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
    )

    sample_frames = []
    target_frames = []
    for split in ("val", "test"):
        sample_df, target_df = audit_split(
            profile_name=args.profile,
            profile=profile,
            split=split,
            lambda_values=np.asarray(lambda_splits[split], dtype=np.float32),
            schedule=schedule,
            lambda_cfg=lambda_cfg,
            args=args,
        )
        sample_frames.append(sample_df)
        target_frames.append(target_df)

    sample_all = pd.concat(sample_frames, ignore_index=True)
    target_all = pd.concat(target_frames, ignore_index=True)
    alpha_csv = Path(args.adaptive_alpha_csv) if args.adaptive_alpha_csv else default_alpha_csv(args.profile)
    target_all = add_alpha_to_target_summary(target_all, alpha_csv)
    align = feature_alignment(sample_all)
    topk = topk_lift(sample_all, parse_float_list(args.top_fracs))
    summary = split_summary(sample_all)

    sample_all.to_csv(out_dir / f"{prefix}_sample_scores.csv", index=False)
    target_all.to_csv(out_dir / f"{prefix}_target_scores.csv", index=False)
    align.to_csv(out_dir / f"{prefix}_feature_alignment.csv", index=False)
    topk.to_csv(out_dir / f"{prefix}_topk_lift.csv", index=False)
    summary.to_csv(out_dir / f"{prefix}_split_summary.csv", index=False)
    pd.DataFrame([lambda_cfg]).to_csv(out_dir / f"{prefix}_lambda_selected.csv", index=False)
    pd.DataFrame([schedule]).to_csv(out_dir / f"{prefix}_schedule_selected.csv", index=False)
    if not args.no_figures:
        write_figures(out_dir, prefix, sample_all, align)
    write_readme(out_dir, prefix, summary, align, topk)
    print(summary.to_string(index=False), flush=True)
    print(f"[Done] outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
