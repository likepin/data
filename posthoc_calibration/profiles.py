from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import diagnose_etth1_lambda_feature_sweep as lambda_sweep
from diagnose_real_lambda_feature_sweep import configure_split, count_csv_rows


REPO = Path(r"C:\Users\cyl\Desktop\iTransformer-phasec-clean")
DATASET_DIR = REPO / "dataset"
RESULT_ROOT = REPO / "results"
DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
AUDIT_ROOT = DATA_ROOT / "deltaA_signal_audit"


PROFILES = {
    "weather96_static": {
        "data_csv": DATASET_DIR / "weather.csv",
        "interface_dir": DATA_ROOT / "interfaces" / "Weather_graph_interface_parcorr",
        "static_pattern": "weather_96_96_staticcausal_softmax_itr3_*projection_*",
        "baseline_pattern": "weather_96_96_baseline_itr3_*projection_*",
        "lambda_dir": AUDIT_ROOT / "weather96_lambda_feature_sweep",
        "lambda_prefix": "weather96_static",
        "out_dir": AUDIT_ROOT / "weather96_closed_loop",
        "split": "custom_ratio",
        "date_col": "date",
        "header_mode": "infer",
        "sep": ",",
    },
    "ecl96_static": {
        "data_csv": DATASET_DIR / "ECL.csv",
        "interface_dir": DATA_ROOT / "interfaces" / "ECL_graph_interface_parcorr",
        "static_pattern": "ecl96_confirm_lr5e4_static_anchor_itr3_*projection_*",
        "baseline_pattern": "ecl96_confirm_lr5e4_baseline_itr3_*projection_*",
        "lambda_dir": AUDIT_ROOT / "ecl96_lambda_feature_sweep_static",
        "lambda_prefix": "ecl96_static",
        "out_dir": AUDIT_ROOT / "ecl96_closed_loop_static",
        "split": "custom_ratio",
        "date_col": "date",
        "header_mode": "infer",
        "sep": ",",
    },
    "traffic96_static": {
        "data_csv": DATASET_DIR / "traffic" / "traffic.csv",
        "interface_dir": DATA_ROOT / "interfaces" / "Traffic_graph_interface_parcorr",
        "static_pattern": "traffic_96_96_staticcausal_softmax_itr3_*projection_*",
        "baseline_pattern": "traffic_96_96_baseline_itr3_*projection_*",
        "lambda_dir": AUDIT_ROOT / "traffic96_lambda_feature_sweep",
        "lambda_prefix": "traffic96_static",
        "out_dir": AUDIT_ROOT / "traffic96_closed_loop",
        "split": "custom_ratio",
        "date_col": "date",
        "header_mode": "infer",
        "sep": ",",
    },
    "solar96_static": {
        "data_csv": DATASET_DIR / "Solar" / "solar_AL.txt",
        "interface_dir": DATA_ROOT / "interfaces" / "Solar_graph_interface_parcorr",
        "static_pattern": "solar_96_96_staticcausal_softmax_itr3_*projection_*",
        "baseline_pattern": "solar_96_96_baseline_itr3_*projection_*",
        "lambda_dir": AUDIT_ROOT / "solar96_lambda_feature_sweep",
        "lambda_prefix": "solar96_static",
        "out_dir": AUDIT_ROOT / "solar96_closed_loop",
        "split": "custom_ratio",
        "date_col": None,
        "header_mode": "none",
        "sep": ",",
    },
}


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _row_value(row: dict, names: tuple[str, ...], default):
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        return value
    return default


def _finite_float(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def selected_lambda_config(profile: dict) -> dict:
    prefix = str(profile["lambda_prefix"])
    path = Path(profile["lambda_dir"]) / f"{prefix}_lambda_feature_sweep_validation_fold_stability.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing validation stability file: {path}")

    df = pd.read_csv(path)
    df["_stable"] = df["stable_candidate"].map(parse_bool) if "stable_candidate" in df else False
    pool = df[df["_stable"]].copy()
    if pool.empty:
        pool = df.copy()

    sort_cols = [c for c in ("stability_score", "fold_spearman_mean", "fold_bucket5_lift_mean") if c in pool]
    if sort_cols:
        pool = pool.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    row = pool.iloc[0].to_dict()
    return lambda_config_from_row(row, source_file=path)


def lambda_config_from_row(row: dict, source_file: Path | str) -> dict:
    return {
        "mode": str(row["mode"]),
        "window": int(row["window"]),
        "k": int(row["k"]),
        "lambda_scale": str(_row_value(row, ("lambda_scale", "val_lambda_scale"), "legacy_clipped")),
        "tail_target_width": _finite_float(
            _row_value(row, ("tail_target_width", "val_tail_target_width"), 0.10),
            0.10,
        ),
        "tail_alpha_min": _finite_float(_row_value(row, ("tail_alpha_min", "val_tail_alpha_min"), 0.02), 0.02),
        "tail_alpha_max": _finite_float(_row_value(row, ("tail_alpha_max", "val_tail_alpha_max"), 0.20), 0.20),
        "stable_candidate": parse_bool(row.get("stable_candidate", row.get("_stable", False))),
        "stability_score": float(row.get("stability_score", np.nan)),
        "fold_spearman_mean": float(row.get("fold_spearman_mean", np.nan)),
        "fold_bucket5_lift_mean": float(row.get("fold_bucket5_lift_mean", np.nan)),
        "source_file": str(source_file),
    }


def lambda_candidate_pool(profile: dict, max_candidates: int) -> pd.DataFrame:
    prefix = str(profile["lambda_prefix"])
    lambda_dir = Path(profile["lambda_dir"])
    stability_path = lambda_dir / f"{prefix}_lambda_feature_sweep_validation_fold_stability.csv"
    val_path = lambda_dir / f"{prefix}_lambda_feature_sweep_val.csv"
    if not stability_path.exists():
        raise FileNotFoundError(f"Missing validation stability file: {stability_path}")

    stability = pd.read_csv(stability_path)
    stability["_stable"] = stability["stable_candidate"].map(parse_bool) if "stable_candidate" in stability else False
    stability["_source_file"] = str(stability_path)
    if val_path.exists():
        val = pd.read_csv(val_path)
        rename = {
            column: f"val_{column}"
            for column in val.columns
            if column not in {"mode", "window", "k"}
        }
        val = val.rename(columns=rename)
        pool = stability.merge(val, on=["mode", "window", "k"], how="left")
    else:
        pool = stability.copy()

    stability_score = _numeric(pool, "stability_score")
    selection_score = _numeric(pool, "val_selection_score")
    val_spearman = _numeric(pool, "val_spearman_mse")
    val_bucket_lift = np.clip(_numeric(pool, "val_bucket5_mse_lift_pct") / 20.0, -1.0, 1.0)
    fold_positive = _numeric(pool, "positive_spearman_fraction")
    val_iqr = np.clip(_numeric(pool, "val_lambda_iqr"), 0.0, 1.0)
    pool["pre_quality_score"] = (
        0.25 * stability_score
        + 0.25 * selection_score
        + 0.20 * val_spearman
        + 0.15 * val_bucket_lift
        + 0.10 * fold_positive
        + 0.05 * val_iqr
    )
    pool = pool.sort_values(
        ["_stable", "pre_quality_score", "stability_score", "fold_spearman_mean"],
        ascending=[False, False, False, False],
    )
    if max_candidates > 0:
        pool = pool.head(int(max_candidates))
    return pool.reset_index(drop=True)


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df:
        return np.zeros((len(df),), dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)


def configure_lambda_profile(profile: dict, seq_len: int, pred_len: int, train_ratio: float) -> None:
    data_csv = Path(profile["data_csv"])
    header_mode = str(profile.get("header_mode", "infer"))
    total_rows = count_csv_rows(data_csv, header_mode=header_mode)
    lambda_sweep.DATA_CSV = data_csv
    lambda_sweep.DATA_DATE_COL = profile.get("date_col", "date")
    lambda_sweep.DATA_HEADER_MODE = header_mode
    lambda_sweep.DATA_SEP = str(profile.get("sep", ","))
    lambda_sweep.SEQ_LEN = int(seq_len)
    lambda_sweep.PRED_LEN = int(pred_len)
    configure_split(
        split=str(profile["split"]),
        total_rows=total_rows,
        train_ratio=float(train_ratio),
        seq_len=int(seq_len),
    )


def configure_lambda_scale(lambda_cfg: dict) -> None:
    lambda_sweep.LAMBDA_SCALE = str(lambda_cfg.get("lambda_scale", "legacy_clipped"))
    lambda_sweep.TAIL_TARGET_WIDTH = _finite_float(lambda_cfg.get("tail_target_width"), 0.10)
    lambda_sweep.TAIL_ALPHA_MIN = _finite_float(lambda_cfg.get("tail_alpha_min"), 0.02)
    lambda_sweep.TAIL_ALPHA_MAX = _finite_float(lambda_cfg.get("tail_alpha_max"), 0.20)


def compute_selected_lambda_splits(
    profile: dict,
    lambda_cfg: dict,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
) -> dict[str, np.ndarray]:
    configure_lambda_profile(profile, seq_len=seq_len, pred_len=pred_len, train_ratio=train_ratio)
    configure_lambda_scale(lambda_cfg)
    full_z = lambda_sweep.load_full_z()
    lambda_t = lambda_sweep.compute_lambda_timeline(
        full_z,
        window=int(lambda_cfg["window"]),
        k=int(lambda_cfg["k"]),
        mode=str(lambda_cfg["mode"]),
    )
    return {
        "val": lambda_sweep.lambda_for_split(lambda_t, "val"),
        "test": lambda_sweep.lambda_for_split(lambda_t, "test"),
    }


def split_sample_start_rows(
    profile: dict,
    split: str,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
) -> np.ndarray:
    configure_lambda_profile(profile, seq_len=seq_len, pred_len=pred_len, train_ratio=train_ratio)
    if split == "val":
        border1, border2 = lambda_sweep.TRAIN_END - lambda_sweep.SEQ_LEN, lambda_sweep.VAL_END
    elif split == "test":
        border1, border2 = lambda_sweep.VAL_END - lambda_sweep.SEQ_LEN, lambda_sweep.TEST_END
    else:
        raise ValueError(split)
    n = border2 - border1 - lambda_sweep.SEQ_LEN - lambda_sweep.PRED_LEN + 1
    return np.arange(border1, border1 + n, dtype=np.int64)


def dynamic_args(profile: dict, split: str, pred_len: int, progress_every: int) -> SimpleNamespace:
    pred_file = "val_pred.npy" if split == "val" else "pred.npy"
    true_file = "val_true.npy" if split == "val" else "true.npy"
    return SimpleNamespace(
        interface_dir=str(profile["interface_dir"]),
        result_root=str(RESULT_ROOT),
        data_csv=str(profile["data_csv"]),
        static_pattern=str(profile["static_pattern"]),
        pred_file=pred_file,
        true_file=true_file,
        eval_split=split,
        schedule_source=split,
        pred_len=int(pred_len),
        n_buckets=5,
        gammas=[],
        linear_gamma_min=0.0,
        linear_gamma_max=0.0,
        linear_q_low=0.0,
        linear_q_high=1.0,
        output_prefix="closed_loop_internal",
        progress_every=int(progress_every),
    )
