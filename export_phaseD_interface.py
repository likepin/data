import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def resolve_lambda_array(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "lambda_t" in arr.files:
            arr = arr["lambda_t"]
        elif "arr_0" in arr.files:
            arr = arr["arr_0"]
        else:
            raise ValueError(f"Expected lambda npz with 'lambda_t' or 'arr_0': {path}")
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return arr


def sanitize_lambda(lambda_array: np.ndarray) -> np.ndarray:
    if np.isinf(lambda_array).any():
        raise ValueError("Lambda contains Inf values.")
    nan_mask = np.isnan(lambda_array)
    if not nan_mask.any():
        return lambda_array.astype(np.float32, copy=False)
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        raise ValueError("Lambda is entirely NaN.")
    return np.interp(
        np.arange(len(lambda_array), dtype=np.float64),
        valid_idx.astype(np.float64),
        lambda_array[valid_idx].astype(np.float64),
    ).astype(np.float32)


def build_valid_starts(intervals, seq_len: int, pred_len: int) -> np.ndarray:
    starts = []
    for start, end in intervals:
        last_start = int(end) - int(seq_len) - int(pred_len)
        if last_start < start:
            continue
        starts.extend(range(int(start), last_start + 1))
    return np.asarray(starts, dtype=np.int64)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def aggregate_lag_graph(lag_graph: np.ndarray) -> np.ndarray:
    if lag_graph.ndim != 3:
        raise ValueError(f"Expected lag graph with shape (L, N, N), got {lag_graph.shape}")
    return lag_graph.sum(axis=0).astype(np.float32)


def main():
    repo_root = Path(__file__).resolve().parent
    desktop_root = repo_root.parent

    parser = argparse.ArgumentParser(description="Export Phase D train-time graph interface artifacts.")
    parser.add_argument("--data-dir", default=str(repo_root / "synthetic_step3_v2"))
    parser.add_argument("--exports-dir", default=str(repo_root / "synthetic_step3_v2" / "exports_step5pp" / "phaseD_interface"))
    parser.add_argument("--phasec-artifacts-dir", default=str(desktop_root / "phaseC_artifacts"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    exports_dir = Path(args.exports_dir).resolve()
    phasec_dir = Path(args.phasec_artifacts_dir).resolve()
    exports_dir.mkdir(parents=True, exist_ok=True)

    split_path = phasec_dir / "phaseC_round1_split.json"
    train_cfg_path = phasec_dir / "phaseC_round1_train_config.json"
    lambda_path = phasec_dir / "lambda_gating_locked.npz"
    a_base_path = data_dir / "A_base.npy"
    support_path = data_dir / "adj_base.npy"
    delta_path = data_dir / "DeltaA.npy"

    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))

    cfg = train_cfg["frozen_round1_training_config"]
    seq_len = int(cfg["seq_len"])
    label_len = int(cfg["label_len"])
    pred_len = int(cfg["pred_len"])
    train_intervals = split["splits"]["train"]["intervals"]
    t_switch = int(split["global_boundaries"]["t_switch"])
    expected_length = int(split["indexing"]["length"])

    lambda_raw = resolve_lambda_array(lambda_path)
    if len(lambda_raw) != expected_length:
        raise ValueError(f"Lambda length mismatch: got {len(lambda_raw)}, expected {expected_length}")
    lambda_clean = sanitize_lambda(lambda_raw)

    a_base_lag = np.load(a_base_path).astype(np.float32)
    support = np.load(support_path).astype(np.uint8)
    delta_lag = np.load(delta_path).astype(np.float32)

    a_base_agg = aggregate_lag_graph(a_base_lag)
    delta_agg = aggregate_lag_graph(delta_lag)
    delta_agg = (delta_agg * support.astype(np.float32)).astype(np.float32)

    window_starts = build_valid_starts(train_intervals, seq_len=seq_len, pred_len=pred_len)
    expected_train_windows = int(train_cfg["validation"]["dataset_smoke_check"]["train_windows"])
    if len(window_starts) != expected_train_windows:
        raise ValueError(
            f"Train window count mismatch: got {len(window_starts)}, "
            f"expected {expected_train_windows}"
        )

    lambda_train = np.zeros(len(window_starts), dtype=np.float32)
    delta_train = np.zeros((len(window_starts),) + delta_agg.shape, dtype=np.float32)
    window_index = []

    sample_id = 0
    for interval_id, (interval_start, interval_end) in enumerate(train_intervals):
        last_start = int(interval_end) - seq_len - pred_len
        if last_start < int(interval_start):
            continue
        interval_local_index = 0
        for s_begin in range(int(interval_start), last_start + 1):
            s_end = s_begin + seq_len
            r_begin = s_end - label_len
            r_end = r_begin + label_len + pred_len

            lambda_start = s_begin
            lambda_end = s_end
            lambda_train[sample_id] = float(lambda_clean[lambda_start:lambda_end].mean())

            if r_end <= t_switch:
                gt_regime = 0
                delta_train[sample_id] = 0.0
            elif s_begin >= t_switch:
                gt_regime = 1
                delta_train[sample_id] = delta_agg
            else:
                raise ValueError(f"Unexpected mixed-regime train window at s_begin={s_begin}, r_end={r_end}")

            window_index.append(
                {
                    "sample_id": sample_id,
                    "split": "train",
                    "interval_id": interval_id,
                    "interval_local_index": interval_local_index,
                    "window_start": s_begin,
                    "s_begin": s_begin,
                    "s_end": s_end,
                    "r_begin": r_begin,
                    "r_end": r_end,
                    "lambda_start": lambda_start,
                    "lambda_end": lambda_end,
                    "gt_regime": gt_regime,
                }
            )

            sample_id += 1
            interval_local_index += 1

    if sample_id != len(window_starts):
        raise ValueError(f"Window export mismatch: built {sample_id}, expected {len(window_starts)}")

    a_base_out = exports_dir / "phaseD_a_base_agg.npy"
    support_out = exports_dir / "phaseD_support.npy"
    index_out = exports_dir / "phaseD_window_index_train.json"
    lambda_out = exports_dir / "phaseD_lambda_train.npy"
    delta_out = exports_dir / "phaseD_deltaA_train.npy"
    manifest_out = exports_dir / "phaseD_interface_manifest.json"

    np.save(a_base_out, a_base_agg)
    np.save(support_out, support)
    np.save(lambda_out, lambda_train)
    np.save(delta_out, delta_train)
    index_out.write_text(json.dumps(window_index, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "artifact_name": "phaseD_train_time_graph_interface",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "synthetic_gt_graph",
        "orientation": "tgt_src",
        "data_dir": str(data_dir),
        "exports_dir": str(exports_dir),
        "phasec_artifacts_dir": str(phasec_dir),
        "source_paths": {
            "split": str(split_path),
            "train_config": str(train_cfg_path),
            "lambda_t": str(lambda_path),
            "a_base_lag": str(a_base_path),
            "support": str(support_path),
            "delta_lag": str(delta_path),
        },
        "source_hashes": {
            "split": sha256_file(split_path),
            "train_config": sha256_file(train_cfg_path),
            "lambda_t": sha256_file(lambda_path),
            "a_base_lag": sha256_file(a_base_path),
            "support": sha256_file(support_path),
            "delta_lag": sha256_file(delta_path),
        },
        "window_geometry": {
            "seq_len": seq_len,
            "label_len": label_len,
            "pred_len": pred_len,
            "num_train_windows": len(window_starts),
            "train_intervals": train_intervals,
        },
        "sample_order_hash": sha256_array(window_starts),
        "lambda_contract": {
            "timeline_source": "lambda_gating_locked",
            "sanitization": "linear_interp_with_edge_value_extrapolation",
            "aggregation": "encoder_history_mean",
            "aggregation_interval": "[s_begin, s_end)",
        },
        "graph_contract": {
            "a_base_export": "sum_over_lags",
            "delta_export": "support_masked_signed_sum_over_lags",
            "support_source": "adj_base.npy",
            "window_assignment": "gt_regime_zero_or_post_delta",
        },
        "output_shapes": {
            "phaseD_a_base_agg": list(a_base_agg.shape),
            "phaseD_support": list(support.shape),
            "phaseD_lambda_train": list(lambda_train.shape),
            "phaseD_deltaA_train": list(delta_train.shape),
            "phaseD_window_index_train_count": len(window_index),
        },
        "output_hashes": {
            "phaseD_a_base_agg": sha256_file(a_base_out),
            "phaseD_support": sha256_file(support_out),
            "phaseD_lambda_train": sha256_file(lambda_out),
            "phaseD_deltaA_train": sha256_file(delta_out),
            "phaseD_window_index_train": sha256_file(index_out),
        },
        "counts": {
            "pre_regime_windows": int(sum(item["gt_regime"] == 0 for item in window_index)),
            "post_regime_windows": int(sum(item["gt_regime"] == 1 for item in window_index)),
        },
    }

    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Exported artifacts to: {exports_dir}")
    print(f"train windows: {len(window_starts)}")
    print(f"lambda_train shape: {lambda_train.shape}")
    print(f"delta_train shape: {delta_train.shape}")
    print(f"pre windows: {manifest['counts']['pre_regime_windows']}, post windows: {manifest['counts']['post_regime_windows']}")


if __name__ == "__main__":
    main()
