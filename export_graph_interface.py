import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from step5pp_utils import compute_lambda_kmeans


def resolve_lambda_array(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "lambda_t" in arr.files:
            arr = arr["lambda_t"]
        elif "arr_0" in arr.files:
            arr = arr["arr_0"]
        else:
            raise ValueError(f"Expected lambda npz with 'lambda_t' or 'arr_0': {path}")
    return np.asarray(arr, dtype=np.float32).reshape(-1)


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def aggregate_lag_graph(lag_graph: np.ndarray) -> np.ndarray:
    if lag_graph.ndim != 3:
        raise ValueError(f"Expected lag graph with shape (L, N, N), got {lag_graph.shape}")
    return lag_graph.sum(axis=0).astype(np.float32)


def write_bundle(
    exports_dir: Path,
    a_base_agg: np.ndarray,
    support: np.ndarray,
    lambda_train: np.ndarray,
    delta_train: np.ndarray,
    window_index: list[dict],
    manifest: dict,
) -> None:
    exports_dir.mkdir(parents=True, exist_ok=True)
    a_base_out = exports_dir / "a_base_agg.npy"
    support_out = exports_dir / "support.npy"
    index_out = exports_dir / "window_index_train.json"
    lambda_out = exports_dir / "lambda_train.npy"
    delta_out = exports_dir / "deltaA_train.npy"
    manifest_out = exports_dir / "interface_manifest.json"

    np.save(a_base_out, a_base_agg.astype(np.float32))
    np.save(support_out, support.astype(np.uint8))
    np.save(lambda_out, lambda_train.astype(np.float32))
    np.save(delta_out, delta_train.astype(np.float32))
    index_out.write_text(json.dumps(window_index, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["exports_dir"] = str(exports_dir)
    manifest["output_shapes"] = {
        "a_base_agg": list(a_base_agg.shape),
        "support": list(support.shape),
        "lambda_train": list(lambda_train.shape),
        "deltaA_train": list(delta_train.shape),
        "window_index_train_count": len(window_index),
    }
    manifest["output_hashes"] = {
        "a_base_agg": sha256_file(a_base_out),
        "support": sha256_file(support_out),
        "lambda_train": sha256_file(lambda_out),
        "deltaA_train": sha256_file(delta_out),
        "window_index_train": sha256_file(index_out),
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_progress(exports_dir: Path, payload: dict) -> None:
    exports_dir.mkdir(parents=True, exist_ok=True)
    progress_out = exports_dir / "interface_progress.json"
    progress_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_partial_bundle(
    exports_dir: Path,
    a_base_agg: np.ndarray | None,
    support: np.ndarray | None,
    lambda_train: np.ndarray,
    delta_train: np.ndarray,
    window_index: list[dict],
    completed_windows: int,
) -> None:
    exports_dir.mkdir(parents=True, exist_ok=True)
    if a_base_agg is not None:
        np.save(exports_dir / "a_base_agg.partial.npy", a_base_agg.astype(np.float32))
    if support is not None:
        np.save(exports_dir / "support.partial.npy", support.astype(np.uint8))
    np.save(exports_dir / "lambda_train.partial.npy", lambda_train.astype(np.float32))
    np.save(exports_dir / "deltaA_train.partial.npy", delta_train.astype(np.float32))
    (exports_dir / "window_index_train.partial.json").write_text(
        json.dumps(window_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_progress(
        exports_dir,
        {
            "status": "running",
            "completed_windows": int(completed_windows),
            "total_windows": int(lambda_train.shape[0]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def export_synthetic_gt(args) -> None:
    repo_root = Path(__file__).resolve().parent
    desktop_root = repo_root.parent

    data_dir = Path(args.data_dir).resolve()
    exports_dir = Path(args.exports_dir).resolve()
    phasec_dir = Path(args.phasec_artifacts_dir).resolve()

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
    delta_agg = (aggregate_lag_graph(delta_lag) * support.astype(np.float32)).astype(np.float32)

    window_starts = []
    lambda_train = []
    delta_train = []
    window_index = []
    sample_id = 0

    for interval_id, (interval_start, interval_end) in enumerate(train_intervals):
        last_start = int(interval_end) - seq_len - pred_len
        interval_local_index = 0
        for s_begin in range(int(interval_start), last_start + 1):
            s_end = s_begin + seq_len
            r_begin = s_end - label_len
            r_end = r_begin + label_len + pred_len
            lambda_train.append(float(lambda_clean[s_begin:s_end].mean()))
            if r_end <= t_switch:
                gt_regime = 0
                delta_train.append(np.zeros_like(delta_agg, dtype=np.float32))
            elif s_begin >= t_switch:
                gt_regime = 1
                delta_train.append(delta_agg)
            else:
                raise ValueError(f"Unexpected mixed-regime train window at s_begin={s_begin}, r_end={r_end}")
            window_starts.append(s_begin)
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
                    "lambda_start": s_begin,
                    "lambda_end": s_end,
                    "gt_regime": gt_regime,
                }
            )
            sample_id += 1
            interval_local_index += 1

    lambda_train = np.asarray(lambda_train, dtype=np.float32)
    delta_train = np.asarray(delta_train, dtype=np.float32)
    window_starts = np.asarray(window_starts, dtype=np.int64)

    manifest = {
        "artifact_name": "graph_interface",
        "backend": "synthetic_gt_graph",
        "orientation": "tgt_src",
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
            "num_train_windows": int(len(window_starts)),
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
        "counts": {
            "pre_regime_windows": int(sum(item["gt_regime"] == 0 for item in window_index)),
            "post_regime_windows": int(sum(item["gt_regime"] == 1 for item in window_index)),
        },
    }

    write_bundle(exports_dir, a_base_agg, support, lambda_train, delta_train, window_index, manifest)
    print(f"Exported synthetic_gt graph interface to: {exports_dir}")
    print(f"train windows: {len(window_starts)}")
    print(f"lambda_train shape: {lambda_train.shape}")
    print(f"deltaA_train shape: {delta_train.shape}")


def fit_ridge_with_intercept(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n_samples, n_features = x.shape
    if n_features == 0:
        return np.zeros((0,), dtype=np.float32)
    x_aug = np.concatenate([np.ones((n_samples, 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    reg = np.eye(n_features + 1, dtype=np.float64)
    reg[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + alpha * reg
    rhs = x_aug.T @ y.astype(np.float64)
    try:
        coef_aug = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef_aug = np.linalg.pinv(lhs) @ rhs
    return coef_aug[1:].astype(np.float32)


def _resolve_value_columns(df: pd.DataFrame, date_col: str | None, value_cols: list[str] | None) -> list[str]:
    if value_cols:
        missing = [col for col in value_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Requested value columns missing from CSV: {missing}")
        return value_cols
    return [col for col in df.columns if col != date_col]


def _resolve_train_end(total_rows: int, split_mode: str, train_end: int | None, train_ratio: float | None) -> int:
    if split_mode == "ett_hour":
        return 12 * 30 * 24
    if split_mode == "ett_minute":
        return 12 * 30 * 24 * 4
    if split_mode == "train_end":
        if train_end is None:
            raise ValueError("--train-end is required when split-mode=train_end")
        return int(train_end)
    if split_mode == "custom_ratio":
        if train_ratio is None:
            raise ValueError("--train-ratio is required when split-mode=custom_ratio")
        return int(total_rows * float(train_ratio))
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def build_real_train_standardized(
    data_path: Path,
    dataset_name: str,
    split_mode: str,
    train_end: int | None,
    train_ratio: float | None,
    date_col: str | None,
    value_cols: list[str] | None,
) -> tuple[np.ndarray, dict]:
    df = pd.read_csv(data_path)
    resolved_date_col = None
    if date_col and date_col in df.columns:
        resolved_date_col = date_col
    elif date_col:
        raise ValueError(f"date column not found: {date_col}")
    resolved_value_cols = _resolve_value_columns(df, resolved_date_col, value_cols)
    values = df[resolved_value_cols].to_numpy(dtype=np.float64)
    resolved_train_end = _resolve_train_end(
        total_rows=len(values),
        split_mode=split_mode,
        train_end=train_end,
        train_ratio=train_ratio,
    )
    if resolved_train_end <= 0 or resolved_train_end > len(values):
        raise ValueError(f"Invalid train_end={resolved_train_end} for total_rows={len(values)}")
    train_values = values[:resolved_train_end]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_z = ((train_values - mean) / std).astype(np.float32)
    meta = {
        "dataset": dataset_name,
        "num_variables": int(train_z.shape[1]),
        "train_end": int(resolved_train_end),
        "total_rows": int(len(values)),
        "columns": list(resolved_value_cols),
        "date_col": resolved_date_col,
        "split_mode": split_mode,
        "train_ratio": None if train_ratio is None else float(train_ratio),
    }
    return train_z, meta


def make_cond_ind_test(args):
    if args.cond_test == "parcorr":
        return ParCorr(significance="analytic"), {
            "cond_test": "ParCorr",
            "significance": "analytic",
        }
    if args.cond_test == "cmiknn":
        try:
            from tigramite.independence_tests.cmiknn import CMIknn
        except ImportError as exc:
            raise ImportError(
                "CMIknn requires optional tigramite dependencies (notably numba). "
                "Install the missing dependency before using --cond-test cmiknn."
            ) from exc
        return CMIknn(
            significance=args.cmiknn_significance,
            knn=int(args.cmiknn_knn),
            shuffle_neighbors=int(args.cmiknn_shuffle_neighbors),
            sig_samples=int(args.cmiknn_sig_samples),
            verbosity=int(args.cmiknn_verbosity),
        ), {
            "cond_test": "CMIknn",
            "significance": args.cmiknn_significance,
            "knn": int(args.cmiknn_knn),
            "shuffle_neighbors": int(args.cmiknn_shuffle_neighbors),
            "sig_samples": int(args.cmiknn_sig_samples),
            "verbosity": int(args.cmiknn_verbosity),
        }
    raise ValueError(f"Unsupported cond_test: {args.cond_test}")


def run_pcmci(
    train_z: np.ndarray,
    tau_max: int,
    pc_alpha: float,
    cond_ind_test,
    pcmci_verbosity: int,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int]]]]:
    pcmci = PCMCI(
        dataframe=DataFrame(train_z.astype(np.float64)),
        cond_ind_test=cond_ind_test,
        verbosity=int(pcmci_verbosity),
    )
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    n_vars = train_z.shape[1]
    a_base_lag = np.zeros((tau_max, n_vars, n_vars), dtype=np.float32)
    support_lag = np.zeros((tau_max, n_vars, n_vars), dtype=np.uint8)
    parents_by_target: list[list[tuple[int, int]]] = [[] for _ in range(n_vars)]
    for src in range(n_vars):
        for tgt in range(n_vars):
            for lag in range(1, tau_max + 1):
                if p_matrix[src, tgt, lag] <= pc_alpha:
                    a_base_lag[lag - 1, tgt, src] = float(val_matrix[src, tgt, lag])
                    support_lag[lag - 1, tgt, src] = 1
                    parents_by_target[tgt].append((src, lag))
    return a_base_lag, support_lag, parents_by_target


def build_global_design(train_z: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
    total_steps, n_vars = train_z.shape
    rows = total_steps - tau_max
    design_all = np.zeros((rows, tau_max * n_vars), dtype=np.float32)
    for lag in range(1, tau_max + 1):
        design_all[:, (lag - 1) * n_vars: lag * n_vars] = train_z[tau_max - lag: total_steps - lag]
    targets_all = train_z[tau_max:].astype(np.float32)
    return design_all, targets_all


def export_real_estimated(args) -> None:
    data_path = Path(args.data_path).resolve()
    exports_dir = Path(args.exports_dir).resolve()
    seq_len = int(args.seq_len)
    label_len = int(args.label_len)
    pred_len = int(args.pred_len)
    tau_max = int(args.tau_max)
    pc_alpha = float(args.pc_alpha)
    ridge_alpha = float(args.ridge_alpha)
    lambda_window = int(args.lambda_window)
    lambda_k = int(args.lambda_k)
    lambda_seed = int(args.lambda_seed)
    progress_every = int(args.progress_every)
    checkpoint_every = int(args.checkpoint_every)
    overall_start = time.time()

    write_progress(
        exports_dir,
        {
            "status": "starting",
            "stage": "load_and_standardize",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source_csv": str(data_path),
        },
    )
    print(f"[Stage] load_and_standardize | source={data_path}")

    stage_start = time.time()
    train_z, meta = build_real_train_standardized(
        data_path=data_path,
        dataset_name=args.dataset_name,
        split_mode=args.split_mode,
        train_end=args.train_end,
        train_ratio=args.train_ratio,
        date_col=args.date_col,
        value_cols=args.value_cols,
    )
    train_length, n_vars = train_z.shape
    print(
        f"[Done] load_and_standardize | rows={train_length} | vars={n_vars} | "
        f"elapsed={(time.time() - stage_start)/60:.2f} min"
    )

    write_progress(
        exports_dir,
        {
            "status": "running",
            "stage": "lambda_kmeans",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "train_length": int(train_length),
            "num_variables": int(n_vars),
        },
    )
    print(
        f"[Stage] lambda_kmeans | window={lambda_window} | k={lambda_k} | seed={lambda_seed}"
    )
    stage_start = time.time()
    lambda_raw, _ = compute_lambda_kmeans(train_z.astype(np.float64), window=lambda_window, k=lambda_k, seed=lambda_seed)
    lambda_clean = sanitize_lambda(np.asarray(lambda_raw, dtype=np.float32))
    print(f"[Done] lambda_kmeans | elapsed={(time.time() - stage_start)/60:.2f} min")

    cond_ind_test, cond_test_meta = make_cond_ind_test(args)
    write_progress(
        exports_dir,
        {
            "status": "running",
            "stage": "pcmci",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "tau_max": tau_max,
            "pc_alpha": pc_alpha,
            "cond_test": cond_test_meta,
        },
    )
    print(
        f"[Stage] pcmci | cond_test={cond_test_meta['cond_test']} | tau_max={tau_max} | "
        f"pc_alpha={pc_alpha}"
    )
    stage_start = time.time()
    a_base_lag, support_lag, parents_by_target = run_pcmci(
        train_z,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        cond_ind_test=cond_ind_test,
        pcmci_verbosity=int(args.pcmci_verbosity),
    )
    print(f"[Done] pcmci | elapsed={(time.time() - stage_start)/60:.2f} min")
    support = (support_lag.sum(axis=0) > 0).astype(np.uint8)
    a_base_agg = aggregate_lag_graph(a_base_lag)
    np.save(exports_dir / "a_base_agg.partial.npy", a_base_agg.astype(np.float32))
    np.save(exports_dir / "support.partial.npy", support.astype(np.uint8))

    design_all, targets_all = build_global_design(train_z, tau_max=tau_max)

    num_windows = train_length - seq_len - pred_len + 1
    if num_windows <= 0:
        raise ValueError("Invalid ETTh1 window geometry; no train windows available.")

    lambda_train = np.zeros((num_windows,), dtype=np.float32)
    delta_train = np.zeros((num_windows, n_vars, n_vars), dtype=np.float32)
    window_index = []
    window_starts = np.arange(num_windows, dtype=np.int64)

    parent_cols_by_target = []
    for tgt in range(n_vars):
        cols = [((lag - 1) * n_vars + src) for src, lag in parents_by_target[tgt]]
        parent_cols_by_target.append(cols)

    write_progress(
        exports_dir,
        {
            "status": "running",
            "stage": "window_ridge",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_windows": int(num_windows),
            "checkpoint_every": checkpoint_every,
            "progress_every": progress_every,
        },
    )
    print(f"[Stage] window_ridge | num_windows={num_windows}")
    stage_start = time.time()
    for sample_id, s_begin in enumerate(range(num_windows)):
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len

        lambda_train[sample_id] = float(lambda_clean[s_begin:s_end].mean())
        row_start = s_begin
        row_end = s_end - tau_max
        local_agg = np.zeros((n_vars, n_vars), dtype=np.float32)

        for tgt in range(n_vars):
            parent_cols = parent_cols_by_target[tgt]
            if not parent_cols:
                continue
            x_win = design_all[row_start:row_end, parent_cols]
            y_win = targets_all[row_start:row_end, tgt]
            coef = fit_ridge_with_intercept(x_win, y_win, alpha=ridge_alpha)
            for weight, (src, lag) in zip(coef, parents_by_target[tgt]):
                local_agg[tgt, src] += weight

        delta_train[sample_id] = (local_agg - a_base_agg) * support.astype(np.float32)
        window_index.append(
            {
                "sample_id": sample_id,
                "split": "train",
                "interval_id": 0,
                "interval_local_index": sample_id,
                "window_start": s_begin,
                "s_begin": s_begin,
                "s_end": s_end,
                "r_begin": r_begin,
                "r_end": r_end,
                "lambda_start": s_begin,
                "lambda_end": s_end,
            }
        )

        completed = sample_id + 1
        if progress_every > 0 and (completed % progress_every == 0 or completed == num_windows):
            elapsed_min = (time.time() - stage_start) / 60.0
            total_elapsed_min = (time.time() - overall_start) / 60.0
            print(
                f"[Progress] window_ridge {completed}/{num_windows} | "
                f"stage_elapsed={elapsed_min:.2f} min | total_elapsed={total_elapsed_min:.2f} min"
            )
        if checkpoint_every > 0 and (completed % checkpoint_every == 0 or completed == num_windows):
            write_partial_bundle(
                exports_dir=exports_dir,
                a_base_agg=a_base_agg,
                support=support,
                lambda_train=lambda_train,
                delta_train=delta_train,
                window_index=window_index,
                completed_windows=completed,
            )

    manifest = {
        "artifact_name": "graph_interface",
        "backend": "real_estimated_graph",
        "orientation": "tgt_src",
        "source_paths": {
            "data_csv": str(data_path),
        },
        "source_hashes": {
            "data_csv": sha256_file(data_path),
        },
        "dataset_contract": {
            "dataset": meta["dataset"],
            "columns": meta["columns"],
            "train_length": int(train_length),
            "num_variables": int(n_vars),
            "value_preprocessing": "train_full_zscore",
            "split_mode": meta["split_mode"],
            "date_col": meta["date_col"],
            "total_rows": meta["total_rows"],
        },
        "window_geometry": {
            "seq_len": seq_len,
            "label_len": label_len,
            "pred_len": pred_len,
            "num_train_windows": int(num_windows),
            "train_interval": [0, int(train_length)],
        },
        "sample_order_hash": sha256_array(window_starts),
        "lambda_contract": {
            "timeline_source": "kmeans_distance_to_cluster_center",
            "timeline_backend": "compute_lambda_kmeans",
            "lambda_window": lambda_window,
            "lambda_k": lambda_k,
            "lambda_seed": lambda_seed,
            "sanitization": "linear_interp_with_edge_value_extrapolation",
            "aggregation": "encoder_history_mean",
            "aggregation_interval": "[s_begin, s_end)",
        },
        "graph_contract": {
            "static_source": "pcmci_train_full",
            "tau_max": tau_max,
            "pc_alpha": pc_alpha,
            "a_base_export": "significant_signed_sum_over_lags",
            "support_export": "collapsed_significant_support_over_lags",
            "local_estimator": "windowed_ridge_on_fixed_pcmci_support",
            "ridge_alpha": ridge_alpha,
            "delta_export": "support_masked_signed_sum_over_lags",
            "cond_test": cond_test_meta,
        },
    }

    write_progress(
        exports_dir,
        {
            "status": "finalizing",
            "stage": "write_bundle",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_windows": int(num_windows),
        },
    )
    write_bundle(exports_dir, a_base_agg, support, lambda_train, delta_train, window_index, manifest)
    write_progress(
        exports_dir,
        {
            "status": "completed",
            "stage": "done",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_windows": int(num_windows),
            "elapsed_minutes": round((time.time() - overall_start) / 60.0, 4),
        },
    )
    print(f"Exported real_estimated graph interface to: {exports_dir}")
    print(f"train windows: {num_windows}")
    print(f"lambda_train shape: {lambda_train.shape}")
    print(f"deltaA_train shape: {delta_train.shape}")


def export_etth1_estimated(args) -> None:
    args.dataset_name = "ETTh1"
    args.split_mode = "ett_hour"
    args.train_end = None
    args.train_ratio = None
    args.date_col = "date"
    args.value_cols = None
    export_real_estimated(args)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    desktop_root = repo_root.parent

    parser = argparse.ArgumentParser(description="Export generic train-time graph interface artifacts.")
    subparsers = parser.add_subparsers(dest="backend", required=True)

    synthetic_parser = subparsers.add_parser("synthetic_gt")
    synthetic_parser.add_argument("--data-dir", default=str(repo_root / "synthetic_step3_v2"))
    synthetic_parser.add_argument("--exports-dir", default=str(repo_root / "synthetic_step3_v2" / "exports_step5pp" / "graph_interface"))
    synthetic_parser.add_argument("--phasec-artifacts-dir", default=str(desktop_root / "phaseC_artifacts"))

    real_parser = subparsers.add_parser("real_estimated")
    real_parser.add_argument("--dataset-name", default="real_dataset")
    real_parser.add_argument("--data-path", required=True)
    real_parser.add_argument("--exports-dir", required=True)
    real_parser.add_argument("--date-col", default="date")
    real_parser.add_argument("--value-cols", nargs="*", default=None)
    real_parser.add_argument("--split-mode", choices=["ett_hour", "ett_minute", "custom_ratio", "train_end"], default="custom_ratio")
    real_parser.add_argument("--train-end", type=int, default=None)
    real_parser.add_argument("--train-ratio", type=float, default=0.7)
    real_parser.add_argument("--seq-len", type=int, default=96)
    real_parser.add_argument("--label-len", type=int, default=48)
    real_parser.add_argument("--pred-len", type=int, default=96)
    real_parser.add_argument("--tau-max", type=int, default=2)
    real_parser.add_argument("--pc-alpha", type=float, default=0.01)
    real_parser.add_argument("--ridge-alpha", type=float, default=1.0)
    real_parser.add_argument("--lambda-window", type=int, default=40)
    real_parser.add_argument("--lambda-k", type=int, default=2)
    real_parser.add_argument("--lambda-seed", type=int, default=2023)
    real_parser.add_argument("--cond-test", choices=["parcorr", "cmiknn"], default="cmiknn")
    real_parser.add_argument("--cmiknn-significance", choices=["shuffle_test"], default="shuffle_test")
    real_parser.add_argument("--cmiknn-knn", type=int, default=20)
    real_parser.add_argument("--cmiknn-shuffle-neighbors", type=int, default=10)
    real_parser.add_argument("--cmiknn-sig-samples", type=int, default=200)
    real_parser.add_argument("--cmiknn-verbosity", type=int, default=0)
    real_parser.add_argument("--pcmci-verbosity", type=int, default=1)
    real_parser.add_argument("--progress-every", type=int, default=250)
    real_parser.add_argument("--checkpoint-every", type=int, default=500)

    etth1_parser = subparsers.add_parser("etth1_estimated")
    etth1_parser.add_argument("--data-path", default=str(desktop_root / "ETDataset-main" / "ETT-small" / "ETTh1.csv"))
    etth1_parser.add_argument("--exports-dir", default=str(repo_root / "interfaces" / "ETTh1_graph_interface_cmiknn"))
    etth1_parser.add_argument("--seq-len", type=int, default=96)
    etth1_parser.add_argument("--label-len", type=int, default=48)
    etth1_parser.add_argument("--pred-len", type=int, default=96)
    etth1_parser.add_argument("--tau-max", type=int, default=2)
    etth1_parser.add_argument("--pc-alpha", type=float, default=0.01)
    etth1_parser.add_argument("--ridge-alpha", type=float, default=1.0)
    etth1_parser.add_argument("--lambda-window", type=int, default=40)
    etth1_parser.add_argument("--lambda-k", type=int, default=2)
    etth1_parser.add_argument("--lambda-seed", type=int, default=2023)
    etth1_parser.add_argument("--cond-test", choices=["parcorr", "cmiknn"], default="cmiknn")
    etth1_parser.add_argument("--cmiknn-significance", choices=["shuffle_test"], default="shuffle_test")
    etth1_parser.add_argument("--cmiknn-knn", type=int, default=20)
    etth1_parser.add_argument("--cmiknn-shuffle-neighbors", type=int, default=10)
    etth1_parser.add_argument("--cmiknn-sig-samples", type=int, default=200)
    etth1_parser.add_argument("--cmiknn-verbosity", type=int, default=0)
    etth1_parser.add_argument("--pcmci-verbosity", type=int, default=1)
    etth1_parser.add_argument("--progress-every", type=int, default=250)
    etth1_parser.add_argument("--checkpoint-every", type=int, default=500)

    args = parser.parse_args()
    if args.backend == "synthetic_gt":
        export_synthetic_gt(args)
    elif args.backend == "real_estimated":
        export_real_estimated(args)
    elif args.backend == "etth1_estimated":
        export_etth1_estimated(args)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")


if __name__ == "__main__":
    main()
