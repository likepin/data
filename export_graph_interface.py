import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from real_dataset_io import load_values_matrix, normalize_date_col, normalize_header_mode

from step5pp_utils import (
    build_window_features,
    kmeans_simple,
    kmeans_sklearn,
    nearest_center_distance,
)


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
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def aggregate_lag_graph(lag_graph: np.ndarray) -> np.ndarray:
    if lag_graph.ndim != 3:
        raise ValueError(f"Expected lag graph with shape (L, N, N), got {lag_graph.shape}")
    return lag_graph.sum(axis=0).astype(np.float32)


def _split_array_path(exports_dir: Path, stem: str, split_name: str, suffix: str = ".npy") -> Path:
    return exports_dir / f"{stem}_{split_name}{suffix}"


def _split_json_path(exports_dir: Path, stem: str, split_name: str) -> Path:
    return exports_dir / f"{stem}_{split_name}.json"


def _partial_array_path(exports_dir: Path, stem: str, suffix: str = ".npy") -> Path:
    return exports_dir / f"{stem}.partial{suffix}"


def _partial_json_path(exports_dir: Path, stem: str) -> Path:
    return exports_dir / f"{stem}.partial.json"


def _is_memmap_array(arr: np.ndarray) -> bool:
    return isinstance(arr, np.memmap)


def _flush_array(arr: np.ndarray) -> None:
    flush = getattr(arr, "flush", None)
    if callable(flush):
        flush()


def _release_memmap_array(arr: np.ndarray) -> None:
    if not _is_memmap_array(arr):
        return
    _flush_array(arr)
    mmap_obj = getattr(arr, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _create_partial_memmap(
    exports_dir: Path,
    stem: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type[np.floating] | type[np.integer],
) -> np.memmap:
    path = _partial_array_path(exports_dir, stem)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(dtype), shape=shape)
    arr[...] = 0
    arr.flush()
    return arr


def _promote_partial_array_to_final(
    *,
    exports_dir: Path,
    stem: str,
    split_name: str,
    values: np.ndarray,
    final_path: Path,
    dtype: np.dtype | type[np.floating] | type[np.integer],
) -> None:
    partial_path = _partial_array_path(exports_dir, f"{stem}_{split_name}")
    if _is_memmap_array(values):
        _release_memmap_array(values)
    if partial_path.exists():
        if final_path.exists():
            final_path.unlink()
        try:
            os.link(partial_path, final_path)
        except OSError:
            partial_path.replace(final_path)
        return
    np.save(final_path, np.asarray(values, dtype=dtype))


def write_bundle(
    exports_dir: Path,
    a_base_agg: np.ndarray,
    support: np.ndarray,
    lambda_train: np.ndarray,
    delta_train: np.ndarray,
    window_index: list[dict],
    manifest: dict,
    extra_split_bundles: dict[str, dict] | None = None,
) -> None:
    exports_dir.mkdir(parents=True, exist_ok=True)
    a_base_out = exports_dir / "a_base_agg.npy"
    support_out = exports_dir / "support.npy"
    manifest_out = exports_dir / "interface_manifest.json"

    np.save(a_base_out, a_base_agg.astype(np.float32))
    np.save(support_out, support.astype(np.uint8))

    split_bundles: dict[str, dict] = {
        "train": {
            "lambda_values": lambda_train,
            "delta_values": delta_train,
            "window_index": window_index,
        }
    }
    if extra_split_bundles:
        for split_name, payload in extra_split_bundles.items():
            if split_name == "train":
                continue
            split_bundles[split_name] = {
                "lambda_values": payload["lambda_values"],
                "delta_values": payload["delta_values"],
                "window_index": list(payload["window_index"]),
            }

    output_shapes = {
        "a_base_agg": list(a_base_agg.shape),
        "support": list(support.shape),
    }
    output_hashes = {
        "a_base_agg": sha256_file(a_base_out),
        "support": sha256_file(support_out),
    }

    for split_name, payload in split_bundles.items():
        lambda_out = _split_array_path(exports_dir, "lambda", split_name)
        delta_out = _split_array_path(exports_dir, "deltaA", split_name)
        index_out = _split_json_path(exports_dir, "window_index", split_name)
        lambda_values = payload["lambda_values"]
        delta_values = payload["delta_values"]
        lambda_shape = list(lambda_values.shape)
        delta_shape = list(delta_values.shape)
        _promote_partial_array_to_final(
            exports_dir=exports_dir,
            stem="lambda",
            split_name=split_name,
            values=lambda_values,
            final_path=lambda_out,
            dtype=np.float32,
        )
        _promote_partial_array_to_final(
            exports_dir=exports_dir,
            stem="deltaA",
            split_name=split_name,
            values=delta_values,
            final_path=delta_out,
            dtype=np.float32,
        )
        index_out.write_text(
            json.dumps(payload["window_index"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_shapes[f"lambda_{split_name}"] = lambda_shape
        output_shapes[f"deltaA_{split_name}"] = delta_shape
        output_shapes[f"window_index_{split_name}_count"] = len(payload["window_index"])
        output_hashes[f"lambda_{split_name}"] = sha256_file(lambda_out)
        output_hashes[f"deltaA_{split_name}"] = sha256_file(delta_out)
        output_hashes[f"window_index_{split_name}"] = sha256_file(index_out)

    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["exports_dir"] = str(exports_dir)
    manifest["output_shapes"] = output_shapes
    manifest["output_hashes"] = output_hashes
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_progress(exports_dir: Path, payload: dict) -> None:
    exports_dir.mkdir(parents=True, exist_ok=True)
    progress_out = exports_dir / "interface_progress.json"
    progress_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_partial_bundle(
    exports_dir: Path,
    a_base_agg: np.ndarray | None,
    support: np.ndarray | None,
    lambda_values: np.ndarray,
    delta_values: np.ndarray,
    window_index: list[dict] | None,
    completed_windows: int,
    split_name: str = "train",
    checkpoint_every: int | None = None,
    progress_every: int | None = None,
    window_delta_topk: int | None = None,
) -> None:
    exports_dir.mkdir(parents=True, exist_ok=True)
    if a_base_agg is not None:
        np.save(_partial_array_path(exports_dir, "a_base_agg"), a_base_agg.astype(np.float32))
    if support is not None:
        np.save(_partial_array_path(exports_dir, "support"), support.astype(np.uint8))
    lambda_path = _partial_array_path(exports_dir, f"lambda_{split_name}")
    delta_path = _partial_array_path(exports_dir, f"deltaA_{split_name}")
    if _is_memmap_array(lambda_values):
        _flush_array(lambda_values)
    else:
        np.save(lambda_path, np.asarray(lambda_values, dtype=np.float32))
    if _is_memmap_array(delta_values):
        _flush_array(delta_values)
    else:
        np.save(delta_path, np.asarray(delta_values, dtype=np.float32))
    if window_index is not None:
        _partial_json_path(exports_dir, f"window_index_{split_name}").write_text(
            json.dumps(window_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _partial_json_path(exports_dir, f"window_ridge_{split_name}_state").write_text(
        json.dumps(
            {
                "split": split_name,
                "completed_windows": int(completed_windows),
                "total_windows": int(lambda_values.shape[0]),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    progress_payload = {
        "status": "running",
        "stage": f"window_ridge_{split_name}",
        "split": split_name,
        "completed_windows": int(completed_windows),
        "total_windows": int(lambda_values.shape[0]),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if checkpoint_every is not None:
        progress_payload["checkpoint_every"] = int(checkpoint_every)
    if progress_every is not None:
        progress_payload["progress_every"] = int(progress_every)
    if window_delta_topk is not None:
        progress_payload["window_delta_topk"] = int(window_delta_topk)
    write_progress(
        exports_dir,
        progress_payload,
    )


def build_parents_from_support_lag(support_lag: np.ndarray) -> list[list[tuple[int, int]]]:
    if support_lag.ndim != 3:
        raise ValueError(f"Expected support_lag with shape (tau_max, n_vars, n_vars), got {support_lag.shape}")
    tau_max, n_vars, _ = support_lag.shape
    parents_by_target: list[list[tuple[int, int]]] = [[] for _ in range(n_vars)]
    for src in range(n_vars):
        for tgt in range(n_vars):
            for lag in range(1, tau_max + 1):
                if int(support_lag[lag - 1, tgt, src]) > 0:
                    parents_by_target[tgt].append((int(src), int(lag)))
    return parents_by_target


def _serialize_pcmci_parents(all_parents: dict[int, list[tuple[int, int]]]) -> dict[str, list[list[int]]]:
    return {
        str(int(target)): [[int(src), int(lag)] for src, lag in parents]
        for target, parents in all_parents.items()
    }


def _deserialize_pcmci_parents(
    parents_payload: dict[str, list[list[int]]] | dict[int, list[tuple[int, int]]],
    n_vars: int,
) -> dict[int, list[tuple[int, int]]]:
    all_parents: dict[int, list[tuple[int, int]]] = {target: [] for target in range(n_vars)}
    for raw_target, raw_parents in parents_payload.items():
        target = int(raw_target)
        if target < 0 or target >= n_vars:
            continue
        parsed: list[tuple[int, int]] = []
        for pair in raw_parents:
            if len(pair) != 2:
                continue
            src, lag = int(pair[0]), int(pair[1])
            parsed.append((src, lag))
        all_parents[target] = parsed
    return all_parents


def _pcmci_target_links(
    *,
    n_vars: int,
    tau_max: int,
    target: int,
    base_link_assumptions: dict[int, dict[tuple[int, int], str]] | None,
) -> dict[tuple[int, int], str]:
    if base_link_assumptions is None:
        return {
            (src, -lag): "-?>"
            for src in range(n_vars)
            for lag in range(1, tau_max + 1)
        }
    return dict(base_link_assumptions.get(target, {}))


def _pcmci_single_target_link_assumptions(
    *,
    n_vars: int,
    tau_max: int,
    target: int,
    base_link_assumptions: dict[int, dict[tuple[int, int], str]] | None,
) -> dict[int, dict[tuple[int, int], str]]:
    target_links = _pcmci_target_links(
        n_vars=n_vars,
        tau_max=tau_max,
        target=target,
        base_link_assumptions=base_link_assumptions,
    )
    link_assumptions = {j: {} for j in range(n_vars)}
    link_assumptions[target] = target_links
    return link_assumptions


def _convert_pcmci_parents_for_ridge(
    all_parents: dict[int, list[tuple[int, int]]],
    n_vars: int,
) -> list[list[tuple[int, int]]]:
    parents_by_target: list[list[tuple[int, int]]] = [[] for _ in range(n_vars)]
    for target in range(n_vars):
        parsed: list[tuple[int, int]] = []
        for src, lag in all_parents.get(target, []):
            if lag >= 0:
                continue
            parsed.append((int(src), int(-lag)))
        parents_by_target[target] = parsed
    return parents_by_target


def save_pcmci_pc_partial(
    exports_dir: Path,
    *,
    n_vars: int,
    tau_max: int,
    all_parents: dict[int, list[tuple[int, int]]],
    completed_targets: list[int],
) -> None:
    _partial_json_path(exports_dir, "pcmci_pc_state").write_text(
        json.dumps(
            {
                "n_vars": int(n_vars),
                "tau_max": int(tau_max),
                "completed_targets": [int(target) for target in completed_targets],
                "all_parents": _serialize_pcmci_parents(all_parents),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_pcmci_pc_partial(exports_dir: Path, *, tau_max: int, n_vars: int) -> dict | None:
    state_path = _partial_json_path(exports_dir, "pcmci_pc_state")
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[Resume] pcmci_pc partial ignored due to invalid state json: {state_path}")
        return None

    if int(state.get("tau_max", -1)) != int(tau_max) or int(state.get("n_vars", -1)) != int(n_vars):
        print(
            f"[Resume] pcmci_pc partial ignored due to metadata mismatch: "
            f"tau_max={state.get('tau_max')} n_vars={state.get('n_vars')} "
            f"expected_tau_max={tau_max} expected_n_vars={n_vars}"
        )
        return None

    try:
        all_parents = _deserialize_pcmci_parents(state.get("all_parents", {}), n_vars=n_vars)
        completed_targets = sorted({int(target) for target in state.get("completed_targets", [])})
    except (TypeError, ValueError):
        print(f"[Resume] pcmci_pc partial ignored due to invalid parents payload: {state_path}")
        return None

    completed_targets = [target for target in completed_targets if 0 <= target < n_vars]
    return {
        "all_parents": all_parents,
        "completed_targets": completed_targets,
    }


def save_pcmci_mci_partial(
    exports_dir: Path,
    *,
    support_lag: np.ndarray,
    n_vars: int,
    tau_max: int,
    completed_targets: list[int],
) -> None:
    np.save(_partial_array_path(exports_dir, "support_lag"), support_lag.astype(np.uint8))
    _partial_json_path(exports_dir, "pcmci_mci_state").write_text(
        json.dumps(
            {
                "n_vars": int(n_vars),
                "tau_max": int(tau_max),
                "completed_targets": [int(target) for target in completed_targets],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_pcmci_mci_partial(exports_dir: Path, *, tau_max: int, n_vars: int) -> dict | None:
    state_path = _partial_json_path(exports_dir, "pcmci_mci_state")
    support_lag_path = _partial_array_path(exports_dir, "support_lag")
    if not state_path.exists() or not support_lag_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[Resume] pcmci_mci partial ignored due to invalid state json: {state_path}")
        return None

    expected_shape = (tau_max, n_vars, n_vars)
    support_lag = np.load(support_lag_path)
    if tuple(support_lag.shape) != expected_shape:
        print(
            f"[Resume] pcmci_mci partial ignored due to shape mismatch: got={tuple(support_lag.shape)} "
            f"expected={expected_shape}"
        )
        return None
    if int(state.get("tau_max", -1)) != int(tau_max) or int(state.get("n_vars", -1)) != int(n_vars):
        print(
            f"[Resume] pcmci_mci partial ignored due to metadata mismatch: "
            f"tau_max={state.get('tau_max')} n_vars={state.get('n_vars')} "
            f"expected_tau_max={tau_max} expected_n_vars={n_vars}"
        )
        return None

    completed_targets = sorted({int(target) for target in state.get("completed_targets", [])})
    completed_targets = [target for target in completed_targets if 0 <= target < n_vars]
    return {
        "support_lag": np.asarray(support_lag, dtype=np.uint8),
        "completed_targets": completed_targets,
    }


def load_pcmci_partial(exports_dir: Path, tau_max: int, n_vars: int) -> np.ndarray | None:
    support_lag_path = _partial_array_path(exports_dir, "support_lag")
    if not support_lag_path.exists():
        return None
    mci_state = load_pcmci_mci_partial(exports_dir, tau_max=tau_max, n_vars=n_vars)
    if mci_state is not None:
        if len(mci_state["completed_targets"]) < n_vars:
            return None
        return np.asarray(mci_state["support_lag"], dtype=np.uint8)
    support_lag = np.load(support_lag_path)
    expected_shape = (tau_max, n_vars, n_vars)
    if tuple(support_lag.shape) != expected_shape:
        print(
            f"[Resume] pcmci partial ignored due to shape mismatch: got={tuple(support_lag.shape)} "
            f"expected={expected_shape}"
        )
        return None
    return np.asarray(support_lag, dtype=np.uint8)


def load_global_ridge_partial(exports_dir: Path, n_vars: int) -> np.ndarray | None:
    a_base_path = _partial_array_path(exports_dir, "a_base_agg")
    if not a_base_path.exists():
        return None
    a_base_agg = np.load(a_base_path)
    expected_shape = (n_vars, n_vars)
    if tuple(a_base_agg.shape) != expected_shape:
        print(
            f"[Resume] global_ridge partial ignored due to shape mismatch: got={tuple(a_base_agg.shape)} "
            f"expected={expected_shape}"
        )
        return None
    return np.asarray(a_base_agg, dtype=np.float32)


def load_window_ridge_partial(exports_dir: Path, split_name: str, num_windows: int, n_vars: int) -> dict | None:
    state_path = _partial_json_path(exports_dir, f"window_ridge_{split_name}_state")
    lambda_path = _partial_array_path(exports_dir, f"lambda_{split_name}")
    delta_path = _partial_array_path(exports_dir, f"deltaA_{split_name}")
    if not state_path.exists() or not lambda_path.exists() or not delta_path.exists():
        return None

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[Resume] window_ridge partial ignored due to invalid state json: {state_path}")
        return None

    lambda_values = np.load(lambda_path, mmap_mode="r+")
    delta_values = np.load(delta_path, mmap_mode="r+")
    expected_lambda_shape = (num_windows,)
    expected_delta_shape = (num_windows, n_vars, n_vars)
    if tuple(lambda_values.shape) != expected_lambda_shape or tuple(delta_values.shape) != expected_delta_shape:
        print(
            f"[Resume] window_ridge partial ignored due to shape mismatch: "
            f"lambda={tuple(lambda_values.shape)} expected={expected_lambda_shape}, "
            f"delta={tuple(delta_values.shape)} expected={expected_delta_shape}"
        )
        return None

    completed_windows = int(state.get("completed_windows", 0))
    if completed_windows < 0 or completed_windows > num_windows:
        print(
            f"[Resume] window_ridge partial ignored due to invalid completed_windows: "
            f"{completed_windows} vs total={num_windows}"
        )
        return None

    return {
        "completed_windows": completed_windows,
        "lambda_values": lambda_values,
        "delta_values": delta_values,
    }


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


def _resolve_model_split_ranges(
    total_rows: int,
    split_mode: str,
    train_end: int | None,
    train_ratio: float | None,
    seq_len: int,
) -> dict[str, dict[str, int]]:
    if split_mode == "ett_hour":
        train_rows = 12 * 30 * 24
        val_rows = 4 * 30 * 24
        test_rows = 4 * 30 * 24
        if train_rows + val_rows + test_rows > total_rows:
            raise ValueError(
                f"ETT-hour split expects at least {train_rows + val_rows + test_rows} rows, got {total_rows}"
            )
        return {
            "train": {"border1": 0, "border2": train_rows},
            "val": {"border1": train_rows - seq_len, "border2": train_rows + val_rows},
            "test": {
                "border1": train_rows + val_rows - seq_len,
                "border2": train_rows + val_rows + test_rows,
            },
        }
    if split_mode == "ett_minute":
        train_rows = 12 * 30 * 24 * 4
        val_rows = 4 * 30 * 24 * 4
        test_rows = 4 * 30 * 24 * 4
        if train_rows + val_rows + test_rows > total_rows:
            raise ValueError(
                f"ETT-minute split expects at least {train_rows + val_rows + test_rows} rows, got {total_rows}"
            )
        return {
            "train": {"border1": 0, "border2": train_rows},
            "val": {"border1": train_rows - seq_len, "border2": train_rows + val_rows},
            "test": {
                "border1": train_rows + val_rows - seq_len,
                "border2": train_rows + val_rows + test_rows,
            },
        }
    if split_mode == "custom_ratio":
        if train_ratio is None:
            raise ValueError("--train-ratio is required when split-mode=custom_ratio")
        num_train = int(total_rows * float(train_ratio))
        num_test = int(total_rows * 0.2)
        num_val = total_rows - num_train - num_test
        if min(num_train, num_val, num_test) <= 0:
            raise ValueError(
                f"Invalid custom split sizes: train={num_train}, val={num_val}, test={num_test}, total={total_rows}"
            )
        return {
            "train": {"border1": 0, "border2": num_train},
            "val": {"border1": num_train - seq_len, "border2": num_train + num_val},
            "test": {"border1": total_rows - num_test - seq_len, "border2": total_rows},
        }
    if split_mode == "train_end":
        raise ValueError(
            "split-mode=train_end is not aligned with iTransformer val/test slicing. "
            "Use split-mode=ett_hour, ett_minute, or custom_ratio for split bundle export."
        )
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def build_real_train_standardized(
    data_path: Path,
    dataset_name: str,
    split_mode: str,
    train_end: int | None,
    train_ratio: float | None,
    date_col: str | None,
    value_cols: list[str] | None,
    header_mode: str | None,
    sep: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    resolved_date_col = normalize_date_col(date_col)
    resolved_header_mode = normalize_header_mode(header_mode)
    values, resolved_value_cols = load_values_matrix(
        data_path=data_path,
        date_col=resolved_date_col,
        value_cols=value_cols,
        header_mode=resolved_header_mode,
        sep=sep,
    )
    total_rows = len(values)
    resolved_train_end = _resolve_train_end(
        total_rows=total_rows,
        split_mode=split_mode,
        train_end=train_end,
        train_ratio=train_ratio,
    )
    split_ranges = _resolve_model_split_ranges(
        total_rows=total_rows,
        split_mode=split_mode,
        train_end=train_end,
        train_ratio=train_ratio,
        seq_len=seq_len,
    )
    train_values = values[:resolved_train_end]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    full_z = ((values - mean) / std).astype(np.float32)
    train_z = full_z[:resolved_train_end]
    meta = {
        "dataset": dataset_name,
        "num_variables": int(train_z.shape[1]),
        "train_end": int(resolved_train_end),
        "total_rows": int(total_rows),
        "columns": list(resolved_value_cols),
        "date_col": resolved_date_col,
        "header_mode": resolved_header_mode,
        "sep": sep,
        "split_mode": split_mode,
        "train_ratio": None if train_ratio is None else float(train_ratio),
        "split_ranges": split_ranges,
    }
    return full_z, train_z, meta


def quantile_normalize_with_reference(
    values: np.ndarray,
    reference_values: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    q10 = np.quantile(reference_values, 0.10)
    q90 = np.quantile(reference_values, 0.90)
    if not np.isfinite(q10) or not np.isfinite(q90) or q90 <= q10 + eps:
        vmin = float(reference_values.min())
        vmax = float(reference_values.max())
        if vmax <= vmin + eps:
            return np.zeros_like(values)
        out = (values - vmin) / (vmax - vmin)
        return np.clip(out, 0.0, 1.0)
    out = (values - q10) / (q90 - q10)
    return np.clip(out, 0.0, 1.0)


def compute_lambda_kmeans_trainfit(
    train_z: np.ndarray,
    full_z: np.ndarray,
    window: int,
    k: int,
    seed: int = 123,
    max_iter: int = 100,
) -> np.ndarray:
    train_feats, _train_idx, _ = build_window_features(train_z, window=window)
    full_feats, full_idx, _ = build_window_features(full_z, window=window)
    train_feats = np.nan_to_num(train_feats, nan=0.0, posinf=0.0, neginf=0.0)
    full_feats = np.nan_to_num(full_feats, nan=0.0, posinf=0.0, neginf=0.0)
    if train_feats.shape[0] == 0 or full_feats.shape[0] == 0:
        raise ValueError("Not enough windows to compute lambda timeline with the requested lambda_window.")

    feat_mean = train_feats.mean(axis=0)
    feat_std = train_feats.std(axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    train_feats_std = (train_feats - feat_mean) / feat_std
    full_feats_std = (full_feats - feat_mean) / feat_std

    km_out = kmeans_sklearn(train_feats_std, k, seed)
    if km_out is None:
        _, centers = kmeans_simple(train_feats_std, k, seed, max_iter=max_iter)
    else:
        _, centers = km_out

    train_dists = nearest_center_distance(train_feats_std, centers)
    full_dists = nearest_center_distance(full_feats_std, centers)
    lambda_valid = quantile_normalize_with_reference(full_dists, train_dists)
    lambda_t = np.full((full_z.shape[0],), np.nan, dtype=np.float64)
    lambda_t[full_idx] = lambda_valid
    return sanitize_lambda(np.asarray(lambda_t, dtype=np.float32))


def make_window_index_record(
    *,
    split_name: str,
    sample_id: int,
    local_start: int,
    border1: int,
    border2: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
) -> dict:
    s_begin = border1 + int(local_start)
    s_end = s_begin + seq_len
    r_begin = s_end - label_len
    r_end = r_begin + label_len + pred_len
    return {
        "sample_id": int(sample_id),
        "split": split_name,
        "interval_id": 0,
        "interval_local_index": int(local_start),
        "window_start": int(local_start),
        "absolute_window_start": int(s_begin),
        "s_begin": int(s_begin),
        "s_end": int(s_end),
        "r_begin": int(r_begin),
        "r_end": int(r_end),
        "lambda_start": int(s_begin),
        "lambda_end": int(s_end),
        "split_border1": int(border1),
        "split_border2": int(border2),
    }


def build_split_window_bundle(
    *,
    split_name: str,
    split_range: dict[str, int],
    seq_len: int,
    label_len: int,
    pred_len: int,
    tau_max: int,
    full_lambda_clean: np.ndarray,
    design_all_full: np.ndarray,
    targets_all_full: np.ndarray,
    parents_by_target: list[list[tuple[int, int]]],
    n_vars: int,
    ridge_alpha: float,
    a_base_agg: np.ndarray,
    support: np.ndarray,
    window_delta_topk: int,
    progress_every: int,
    checkpoint_every: int,
    exports_dir: Path | None = None,
    resume_state: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    border1 = int(split_range["border1"])
    border2 = int(split_range["border2"])
    num_windows = border2 - border1 - seq_len - pred_len + 1
    if num_windows <= 0:
        raise ValueError(
            f"Invalid split geometry for {split_name}: border1={border1}, border2={border2}, "
            f"seq_len={seq_len}, pred_len={pred_len}"
        )

    local_window_starts = np.arange(num_windows, dtype=np.int64)
    completed_windows = 0
    if resume_state is not None:
        lambda_values = resume_state["lambda_values"]
        delta_values = resume_state["delta_values"]
        completed_windows = int(resume_state["completed_windows"])
        print(f"[Resume] window_ridge | split={split_name} | completed_windows={completed_windows}/{num_windows}")
    elif exports_dir is not None:
        lambda_values = _create_partial_memmap(
            exports_dir=exports_dir,
            stem=f"lambda_{split_name}",
            shape=(num_windows,),
            dtype=np.float32,
        )
        delta_values = _create_partial_memmap(
            exports_dir=exports_dir,
            stem=f"deltaA_{split_name}",
            shape=(num_windows, n_vars, n_vars),
            dtype=np.float32,
        )
    else:
        lambda_values = np.zeros((num_windows,), dtype=np.float32)
        delta_values = np.zeros((num_windows, n_vars, n_vars), dtype=np.float32)

    stage_start = time.time()
    support_float = support.astype(np.float32, copy=False)

    for sample_id in range(completed_windows, num_windows):
        local_start = int(local_window_starts[sample_id])
        s_begin = border1 + int(local_start)
        s_end = s_begin + seq_len

        lambda_values[sample_id] = float(full_lambda_clean[s_begin:s_end].mean())
        row_start = s_begin
        row_end = s_end - tau_max
        local_agg = fit_aggregated_ridge_graph(
            design_all=design_all_full,
            targets_all=targets_all_full,
            parents_by_target=parents_by_target,
            n_vars=n_vars,
            ridge_alpha=ridge_alpha,
            row_start=row_start,
            row_end=row_end,
        )
        delta_raw = (local_agg - a_base_agg) * support_float
        delta_values[sample_id] = sparsify_window_delta(delta_raw, topk=window_delta_topk)
        completed_now = sample_id + 1
        if exports_dir is not None and checkpoint_every > 0:
            if completed_now % checkpoint_every == 0 or completed_now == num_windows:
                write_partial_bundle(
                    exports_dir=exports_dir,
                    a_base_agg=None,
                    support=None,
                    lambda_values=lambda_values,
                    delta_values=delta_values,
                    window_index=None,
                    completed_windows=completed_now,
                    split_name=split_name,
                    checkpoint_every=checkpoint_every,
                    progress_every=progress_every,
                    window_delta_topk=window_delta_topk,
                )
        if progress_every > 0 and (completed_now % progress_every == 0 or completed_now == num_windows):
            elapsed_min = (time.time() - stage_start) / 60.0
            print(
                f"[Progress] window_ridge | split={split_name} | completed={completed_now}/{num_windows} "
                f"| elapsed={elapsed_min:.2f} min"
            )
            if exports_dir is not None:
                write_progress(
                    exports_dir,
                    {
                        "status": "running",
                        "stage": f"window_ridge_{split_name}",
                        "split": split_name,
                        "completed_windows": int(completed_now),
                        "total_windows": int(num_windows),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "checkpoint_every": int(checkpoint_every),
                        "progress_every": int(progress_every),
                        "window_delta_topk": int(window_delta_topk),
                    },
                )

    window_index = [
        make_window_index_record(
            split_name=split_name,
            sample_id=sample_id,
            local_start=int(local_start),
            border1=border1,
            border2=border2,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
        )
        for sample_id, local_start in enumerate(local_window_starts)
    ]
    return lambda_values, delta_values, window_index, local_window_starts


def describe_cond_ind_test(args) -> dict:
    if args.cond_test == "parcorr":
        return {
            "cond_test": "ParCorr",
            "significance": "analytic",
        }
    if args.cond_test == "cmiknn":
        return {
            "cond_test": "CMIknn",
            "significance": args.cmiknn_significance,
            "knn": int(args.cmiknn_knn),
            "shuffle_neighbors": int(args.cmiknn_shuffle_neighbors),
            "sig_samples": int(args.cmiknn_sig_samples),
            "verbosity": int(args.cmiknn_verbosity),
        }
    raise ValueError(f"Unsupported cond_test: {args.cond_test}")


def make_cond_ind_test(args):
    cond_test_meta = describe_cond_ind_test(args)
    if args.cond_test == "parcorr":
        return ParCorr(significance="analytic"), cond_test_meta
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
        ), cond_test_meta
    raise ValueError(f"Unsupported cond_test: {args.cond_test}")


def build_lagcorr_link_assumptions(
    train_z: np.ndarray,
    tau_max: int,
    topk: int,
) -> tuple[dict[int, dict[tuple[int, int], str]], dict[str, float | int | str]]:
    n_obs, n_vars = train_z.shape
    if tau_max < 1:
        raise ValueError("tau_max must be >= 1 for lagcorr prefilter")
    if topk <= 0:
        raise ValueError("topk must be > 0 for lagcorr prefilter")
    if n_obs <= tau_max:
        raise ValueError("Not enough observations for lagged correlation prefilter")

    total_candidates = tau_max * n_vars
    topk = min(int(topk), total_candidates)
    score_grid = np.full((n_vars, total_candidates), -np.inf, dtype=np.float32)

    for lag in range(1, tau_max + 1):
        x = train_z[:-lag].astype(np.float64, copy=False)
        y = train_z[lag:].astype(np.float64, copy=False)
        x_center = x - x.mean(axis=0, keepdims=True)
        y_center = y - y.mean(axis=0, keepdims=True)
        x_norm = np.linalg.norm(x_center, axis=0)
        y_norm = np.linalg.norm(y_center, axis=0)
        denom = np.outer(y_norm, x_norm)
        denom[denom < 1e-12] = np.inf
        corr = (y_center.T @ x_center) / denom
        block_start = (lag - 1) * n_vars
        score_grid[:, block_start : block_start + n_vars] = np.abs(corr).astype(np.float32, copy=False)

    link_assumptions: dict[int, dict[tuple[int, int], str]] = {j: {} for j in range(n_vars)}
    candidate_counts: list[int] = []
    for tgt in range(n_vars):
        row = score_grid[tgt]
        active = np.isfinite(row)
        if not np.any(active):
            candidate_counts.append(0)
            continue
        active_count = int(active.sum())
        keep = min(topk, active_count)
        chosen = np.argpartition(row, -keep)[-keep:]
        chosen = chosen[np.argsort(row[chosen])[::-1]]
        for flat_idx in chosen:
            lag = flat_idx // n_vars + 1
            src = flat_idx % n_vars
            link_assumptions[tgt][(src, -lag)] = "-?>"
        candidate_counts.append(int(keep))

    meta = {
        "mode": "lagcorr_topk",
        "topk": int(topk),
        "total_candidates_per_target": int(total_candidates),
        "avg_candidates_per_target": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "max_candidates_per_target": int(max(candidate_counts)) if candidate_counts else 0,
        "min_candidates_per_target": int(min(candidate_counts)) if candidate_counts else 0,
    }
    return link_assumptions, meta


def run_pcmci(
    train_z: np.ndarray,
    tau_max: int,
    pc_alpha: float,
    cond_ind_test,
    pcmci_verbosity: int,
    pcmci_max_conds_dim: int | None = None,
    pcmci_max_conds_py: int | None = None,
    pcmci_max_conds_px: int | None = None,
    link_assumptions: dict[int, dict[tuple[int, int], str]] | None = None,
    exports_dir: Path | None = None,
    progress_every_targets: int = 0,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    n_vars = train_z.shape[1]
    progress_every_targets = max(0, int(progress_every_targets))
    pcmci = PCMCI(
        dataframe=DataFrame(train_z.astype(np.float64)),
        cond_ind_test=cond_ind_test,
        verbosity=int(pcmci_verbosity),
    )

    pc_resume = None
    if exports_dir is not None:
        pc_resume = load_pcmci_pc_partial(exports_dir, tau_max=tau_max, n_vars=n_vars)
    all_parents: dict[int, list[tuple[int, int]]] = {target: [] for target in range(n_vars)}
    completed_pc_targets: set[int] = set()
    if pc_resume is not None:
        all_parents = pc_resume["all_parents"]
        completed_pc_targets = set(pc_resume["completed_targets"])
        print(f"[Resume] pcmci_pc | completed_targets={len(completed_pc_targets)}/{n_vars}")

    pc_stage_start = time.time()
    for target in range(n_vars):
        if target in completed_pc_targets:
            continue
        target_links = _pcmci_target_links(
            n_vars=n_vars,
            tau_max=tau_max,
            target=target,
            base_link_assumptions=link_assumptions,
        )
        pc_single_result = pcmci._run_pc_stable_single(
            j=target,
            link_assumptions_j=target_links,
            tau_min=1,
            tau_max=tau_max,
            save_iterations=False,
            pc_alpha=pc_alpha,
            max_conds_dim=pcmci_max_conds_dim,
            max_combinations=1,
        )
        parents_j = pc_single_result["parents"]
        all_parents[target] = [(int(src), int(lag)) for src, lag in parents_j]
        completed_pc_targets.add(target)
        if exports_dir is not None:
            save_pcmci_pc_partial(
                exports_dir,
                n_vars=n_vars,
                tau_max=tau_max,
                all_parents=all_parents,
                completed_targets=sorted(completed_pc_targets),
            )
        completed_now = len(completed_pc_targets)
        if progress_every_targets > 0 and (completed_now % progress_every_targets == 0 or completed_now == n_vars):
            elapsed_min = (time.time() - pc_stage_start) / 60.0
            print(
                f"[Progress] pcmci_pc | completed_targets={completed_now}/{n_vars} | "
                f"elapsed={elapsed_min:.2f} min"
            )
            if exports_dir is not None:
                write_progress(
                    exports_dir,
                    {
                        "status": "running",
                        "stage": "pcmci_pc",
                        "completed_targets": int(completed_now),
                        "total_targets": int(n_vars),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

    mci_resume = None
    if exports_dir is not None:
        mci_resume = load_pcmci_mci_partial(exports_dir, tau_max=tau_max, n_vars=n_vars)
    support_lag = np.zeros((tau_max, n_vars, n_vars), dtype=np.uint8)
    completed_mci_targets: set[int] = set()
    if mci_resume is not None:
        support_lag = np.asarray(mci_resume["support_lag"], dtype=np.uint8)
        completed_mci_targets = set(mci_resume["completed_targets"])
        print(f"[Resume] pcmci_mci | completed_targets={len(completed_mci_targets)}/{n_vars}")

    mci_stage_start = time.time()
    for target in range(n_vars):
        if target in completed_mci_targets:
            continue
        target_assumptions = _pcmci_single_target_link_assumptions(
            n_vars=n_vars,
            tau_max=tau_max,
            target=target,
            base_link_assumptions=link_assumptions,
        )
        target_links = target_assumptions[target]
        support_lag[:, target, :] = 0
        if target_links:
            results = pcmci.run_mci(
                link_assumptions=target_assumptions,
                tau_min=0,
                tau_max=tau_max,
                parents=all_parents,
                max_conds_py=pcmci_max_conds_py,
                max_conds_px=pcmci_max_conds_px,
                alpha_level=pc_alpha,
                fdr_method="none",
            )
            p_matrix = results["p_matrix"]
            for (src, neg_lag) in target_links.keys():
                lag = int(-neg_lag)
                if p_matrix[int(src), target, lag] <= pc_alpha:
                    support_lag[lag - 1, target, int(src)] = 1
        completed_mci_targets.add(target)
        if exports_dir is not None:
            save_pcmci_mci_partial(
                exports_dir,
                support_lag=support_lag,
                n_vars=n_vars,
                tau_max=tau_max,
                completed_targets=sorted(completed_mci_targets),
            )
        completed_now = len(completed_mci_targets)
        if progress_every_targets > 0 and (completed_now % progress_every_targets == 0 or completed_now == n_vars):
            elapsed_min = (time.time() - mci_stage_start) / 60.0
            print(
                f"[Progress] pcmci_mci | completed_targets={completed_now}/{n_vars} | "
                f"elapsed={elapsed_min:.2f} min"
            )
            if exports_dir is not None:
                write_progress(
                    exports_dir,
                    {
                        "status": "running",
                        "stage": "pcmci_mci",
                        "completed_targets": int(completed_now),
                        "total_targets": int(n_vars),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

    parents_by_target = _convert_pcmci_parents_for_ridge(all_parents, n_vars=n_vars)
    return support_lag, parents_by_target


def build_global_design(train_z: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
    total_steps, n_vars = train_z.shape
    rows = total_steps - tau_max
    design_all = np.zeros((rows, tau_max * n_vars), dtype=np.float32)
    for lag in range(1, tau_max + 1):
        design_all[:, (lag - 1) * n_vars: lag * n_vars] = train_z[tau_max - lag: total_steps - lag]
    targets_all = train_z[tau_max:].astype(np.float32)
    return design_all, targets_all


def fit_aggregated_ridge_graph(
    design_all: np.ndarray,
    targets_all: np.ndarray,
    parents_by_target: list[list[tuple[int, int]]],
    n_vars: int,
    ridge_alpha: float,
    row_start: int,
    row_end: int,
) -> np.ndarray:
    if row_start < 0 or row_end > design_all.shape[0] or row_end < row_start:
        raise ValueError(
            f"Invalid ridge row bounds: row_start={row_start}, row_end={row_end}, design_rows={design_all.shape[0]}"
        )
    agg = np.zeros((n_vars, n_vars), dtype=np.float32)
    for tgt in range(n_vars):
        parents = parents_by_target[tgt]
        if not parents:
            continue
        parent_cols = [((lag - 1) * n_vars + src) for src, lag in parents]
        x_view = design_all[row_start:row_end, parent_cols]
        y_view = targets_all[row_start:row_end, tgt]
        coef = fit_ridge_with_intercept(x_view, y_view, alpha=ridge_alpha)
        for weight, (src, _lag) in zip(coef, parents):
            agg[tgt, src] += weight
    return agg


def derive_window_delta_topk(support: np.ndarray) -> int:
    parent_counts = support.sum(axis=1).astype(np.int64)
    return max(1, int(np.median(parent_counts)))


def sparsify_window_delta(delta_matrix: np.ndarray, topk: int) -> np.ndarray:
    flat = delta_matrix.reshape(-1)
    nonzero_idx = np.flatnonzero(np.abs(flat) > 1e-12)
    if len(nonzero_idx) <= topk:
        return delta_matrix.astype(np.float32, copy=False)
    top_local = np.argpartition(np.abs(flat[nonzero_idx]), -topk)[-topk:]
    keep_idx = nonzero_idx[top_local]
    sparse = np.zeros_like(flat, dtype=np.float32)
    sparse[keep_idx] = flat[keep_idx].astype(np.float32, copy=False)
    return sparse.reshape(delta_matrix.shape)


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
    pcmci_max_conds_dim = args.pcmci_max_conds_dim
    pcmci_max_conds_py = args.pcmci_max_conds_py
    pcmci_max_conds_px = args.pcmci_max_conds_px
    pcmci_prefilter_mode = args.pcmci_prefilter_mode
    pcmci_prefilter_topk = args.pcmci_prefilter_topk
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
    full_z, train_z, meta = build_real_train_standardized(
        data_path=data_path,
        dataset_name=args.dataset_name,
        split_mode=args.split_mode,
        train_end=args.train_end,
        train_ratio=args.train_ratio,
        date_col=args.date_col,
        value_cols=args.value_cols,
        header_mode=args.header_mode,
        sep=args.sep,
        seq_len=seq_len,
    )
    total_rows, n_vars = full_z.shape
    train_length, n_vars = train_z.shape
    print(
        f"[Done] load_and_standardize | train_rows={train_length} | total_rows={total_rows} | vars={n_vars} | "
        f"elapsed={(time.time() - stage_start)/60:.2f} min"
    )

    write_progress(
        exports_dir,
        {
            "status": "running",
            "stage": "lambda_kmeans",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "train_length": int(train_length),
            "total_rows": int(total_rows),
            "num_variables": int(n_vars),
        },
    )
    print(
        f"[Stage] lambda_kmeans | window={lambda_window} | k={lambda_k} | seed={lambda_seed}"
    )
    stage_start = time.time()
    lambda_clean = compute_lambda_kmeans_trainfit(
        train_z=train_z.astype(np.float64),
        full_z=full_z.astype(np.float64),
        window=lambda_window,
        k=lambda_k,
        seed=lambda_seed,
    )
    print(f"[Done] lambda_kmeans | elapsed={(time.time() - stage_start)/60:.2f} min")

    cond_test_meta = describe_cond_ind_test(args)
    pcmci_prefilter_meta = None
    link_assumptions = None
    if pcmci_prefilter_mode == "lagcorr_topk":
        if pcmci_prefilter_topk is None:
            raise ValueError("--pcmci-prefilter-topk is required when --pcmci-prefilter-mode=lagcorr_topk")
        write_progress(
            exports_dir,
            {
                "status": "running",
                "stage": "pcmci_prefilter",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "pcmci_prefilter_mode": pcmci_prefilter_mode,
                "pcmci_prefilter_topk": int(pcmci_prefilter_topk),
            },
        )
        print(f"[Stage] pcmci_prefilter | mode={pcmci_prefilter_mode} | topk={pcmci_prefilter_topk}")
        stage_start = time.time()
        link_assumptions, pcmci_prefilter_meta = build_lagcorr_link_assumptions(
            train_z=train_z,
            tau_max=tau_max,
            topk=int(pcmci_prefilter_topk),
        )
        print(
            f"[Done] pcmci_prefilter | avg_candidates={pcmci_prefilter_meta['avg_candidates_per_target']:.2f} | "
            f"elapsed={(time.time() - stage_start)/60:.2f} min"
        )
    support_lag = load_pcmci_partial(exports_dir, tau_max=tau_max, n_vars=n_vars)
    if support_lag is not None:
        parents_by_target = build_parents_from_support_lag(support_lag)
        print(
            f"[Resume] pcmci | loaded={_partial_array_path(exports_dir, 'support_lag')} | "
            f"cond_test={cond_test_meta['cond_test']}"
        )
    else:
        cond_ind_test, cond_test_meta = make_cond_ind_test(args)
        write_progress(
            exports_dir,
            {
                "status": "running",
                "stage": "pcmci",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "tau_max": tau_max,
                "pc_alpha": pc_alpha,
                "pcmci_max_conds_dim": pcmci_max_conds_dim,
                "pcmci_max_conds_py": pcmci_max_conds_py,
                "pcmci_max_conds_px": pcmci_max_conds_px,
                "pcmci_prefilter": pcmci_prefilter_meta,
                "cond_test": cond_test_meta,
            },
        )
        print(
            f"[Stage] pcmci | cond_test={cond_test_meta['cond_test']} | tau_max={tau_max} | "
            f"pc_alpha={pc_alpha} | max_conds_dim={pcmci_max_conds_dim} | "
            f"max_conds_py={pcmci_max_conds_py} | max_conds_px={pcmci_max_conds_px} | "
            f"prefilter={pcmci_prefilter_meta}"
        )
        stage_start = time.time()
        support_lag, parents_by_target = run_pcmci(
            train_z,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            cond_ind_test=cond_ind_test,
            pcmci_verbosity=int(args.pcmci_verbosity),
            pcmci_max_conds_dim=pcmci_max_conds_dim,
            pcmci_max_conds_py=pcmci_max_conds_py,
            pcmci_max_conds_px=pcmci_max_conds_px,
            link_assumptions=link_assumptions,
            exports_dir=exports_dir,
            progress_every_targets=progress_every,
        )
        print(f"[Done] pcmci | elapsed={(time.time() - stage_start)/60:.2f} min")
        np.save(_partial_array_path(exports_dir, "support_lag"), support_lag.astype(np.uint8))
    support = (support_lag.sum(axis=0) > 0).astype(np.uint8)
    np.save(_partial_array_path(exports_dir, "support"), support.astype(np.uint8))
    window_delta_topk = derive_window_delta_topk(support)

    design_all_train, targets_all_train = build_global_design(train_z, tau_max=tau_max)
    design_all_full, targets_all_full = build_global_design(full_z, tau_max=tau_max)
    a_base_agg = load_global_ridge_partial(exports_dir, n_vars=n_vars)
    if a_base_agg is not None:
        print(f"[Resume] global_ridge | loaded={_partial_array_path(exports_dir, 'a_base_agg')}")
    else:
        write_progress(
            exports_dir,
            {
                "status": "running",
                "stage": "global_ridge",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "ridge_alpha": ridge_alpha,
                "train_design_rows": int(design_all_train.shape[0]),
            },
        )
        print(f"[Stage] global_ridge | ridge_alpha={ridge_alpha}")
        stage_start = time.time()
        a_base_agg = fit_aggregated_ridge_graph(
            design_all=design_all_train,
            targets_all=targets_all_train,
            parents_by_target=parents_by_target,
            n_vars=n_vars,
            ridge_alpha=ridge_alpha,
            row_start=0,
            row_end=design_all_train.shape[0],
        )
        a_base_agg = (a_base_agg * support.astype(np.float32)).astype(np.float32)
        print(f"[Done] global_ridge | elapsed={(time.time() - stage_start)/60:.2f} min")
        np.save(_partial_array_path(exports_dir, "a_base_agg"), a_base_agg.astype(np.float32))

    split_ranges = meta["split_ranges"]
    split_payloads: dict[str, dict] = {}
    sample_order_hash_by_split: dict[str, str] = {}
    split_window_counts: dict[str, int] = {}

    for split_name in ("train", "val", "test"):
        split_range = split_ranges[split_name]
        split_window_count = split_range["border2"] - split_range["border1"] - seq_len - pred_len + 1
        write_progress(
            exports_dir,
            {
                "status": "running",
                "stage": f"window_ridge_{split_name}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "split": split_name,
                "total_windows": int(split_window_count),
                "checkpoint_every": checkpoint_every,
                "progress_every": progress_every,
                "window_delta_topk": int(window_delta_topk),
            },
        )
        print(
            f"[Stage] window_ridge | split={split_name} | num_windows={split_window_count} | "
            f"window_delta_topk={window_delta_topk}"
        )
        stage_start = time.time()
        resume_state = None
        if exports_dir is not None:
            resume_state = load_window_ridge_partial(
                exports_dir=exports_dir,
                split_name=split_name,
                num_windows=split_window_count,
                n_vars=n_vars,
            )
        lambda_values, delta_values, window_index_values, local_window_starts = build_split_window_bundle(
            split_name=split_name,
            split_range=split_range,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            tau_max=tau_max,
            full_lambda_clean=lambda_clean,
            design_all_full=design_all_full,
            targets_all_full=targets_all_full,
            parents_by_target=parents_by_target,
            n_vars=n_vars,
            ridge_alpha=ridge_alpha,
            a_base_agg=a_base_agg,
            support=support,
            window_delta_topk=window_delta_topk,
            progress_every=progress_every,
            checkpoint_every=checkpoint_every,
            exports_dir=exports_dir,
            resume_state=resume_state,
        )
        split_payloads[split_name] = {
            "lambda_values": lambda_values,
            "delta_values": delta_values,
            "window_index": window_index_values,
        }
        sample_order_hash_by_split[split_name] = sha256_array(local_window_starts)
        split_window_counts[split_name] = int(lambda_values.shape[0])
        print(f"[Done] window_ridge | split={split_name} | elapsed={(time.time() - stage_start)/60:.2f} min")
        if exports_dir is not None:
            write_partial_bundle(
                exports_dir=exports_dir,
                a_base_agg=a_base_agg if split_name == "train" else None,
                support=support if split_name == "train" else None,
                lambda_values=lambda_values,
                delta_values=delta_values,
                window_index=window_index_values,
                completed_windows=len(window_index_values),
                split_name=split_name,
                checkpoint_every=checkpoint_every,
                progress_every=progress_every,
                window_delta_topk=window_delta_topk,
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
            "total_rows": int(total_rows),
            "num_variables": int(n_vars),
            "value_preprocessing": "train_full_zscore",
            "split_mode": meta["split_mode"],
            "date_col": meta["date_col"],
            "header_mode": meta["header_mode"],
            "sep": meta["sep"],
            "train_ratio": meta["train_ratio"],
        },
        "window_geometry": {
            "seq_len": seq_len,
            "label_len": label_len,
            "pred_len": pred_len,
            "num_train_windows": int(split_window_counts["train"]),
            "num_val_windows": int(split_window_counts["val"]),
            "num_test_windows": int(split_window_counts["test"]),
            "split_ranges": split_ranges,
            "train_interval": [0, int(train_length)],
        },
        "sample_order_hash": sample_order_hash_by_split["train"],
        "sample_order_hash_by_split": sample_order_hash_by_split,
        "lambda_contract": {
            "timeline_source": "kmeans_distance_to_cluster_center",
            "timeline_backend": "compute_lambda_kmeans_trainfit",
            "lambda_window": lambda_window,
            "lambda_k": lambda_k,
            "lambda_seed": lambda_seed,
            "sanitization": "linear_interp_with_edge_value_extrapolation",
            "aggregation": "encoder_history_mean",
            "aggregation_interval": "[s_begin, s_end)",
            "fit_domain": "train_only",
            "score_domain": "full_timeline",
            "normalization_reference": "train_window_distance_quantiles",
        },
        "graph_contract": {
            "support_source": "pcmci_train_full",
            "static_source": "full_train_ridge_on_pcmci_support",
            "tau_max": tau_max,
            "pc_alpha": pc_alpha,
            "pcmci_max_conds_dim": pcmci_max_conds_dim,
            "pcmci_max_conds_py": pcmci_max_conds_py,
            "pcmci_max_conds_px": pcmci_max_conds_px,
            "pcmci_prefilter": pcmci_prefilter_meta,
            "a_base_export": "full_train_ridge_coeff_sum_over_lags_on_fixed_support",
            "support_export": "collapsed_significant_support_over_lags",
            "local_estimator": "windowed_ridge_on_fixed_pcmci_support",
            "ridge_alpha": ridge_alpha,
            "delta_export": "support_masked_signed_windowed_ridge_minus_global_ridge_topk_sparse",
            "delta_definition": "local_ridge_minus_global_ridge",
            "window_delta_topk_rule": "median_active_parents_per_target_in_support",
            "window_delta_topk": int(window_delta_topk),
            "local_context_domain": "full_standardized_timeline_on_fixed_support",
            "cond_test": cond_test_meta,
        },
    }

    write_progress(
        exports_dir,
        {
            "status": "finalizing",
            "stage": "write_bundle",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_windows": int(sum(split_window_counts.values())),
        },
    )
    write_bundle(
        exports_dir,
        a_base_agg,
        support,
        split_payloads["train"]["lambda_values"],
        split_payloads["train"]["delta_values"],
        split_payloads["train"]["window_index"],
        manifest,
        extra_split_bundles={
            "val": split_payloads["val"],
            "test": split_payloads["test"],
        },
    )
    write_progress(
        exports_dir,
        {
            "status": "completed",
            "stage": "done",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_windows": int(sum(split_window_counts.values())),
            "elapsed_minutes": round((time.time() - overall_start) / 60.0, 4),
        },
    )
    print(f"Exported real_estimated graph interface to: {exports_dir}")
    print(f"train windows: {split_window_counts['train']}")
    print(f"val windows: {split_window_counts['val']}")
    print(f"test windows: {split_window_counts['test']}")
    print(f"lambda_train shape: {split_payloads['train']['lambda_values'].shape}")
    print(f"deltaA_train shape: {split_payloads['train']['delta_values'].shape}")
    print(f"lambda_val shape: {split_payloads['val']['lambda_values'].shape}")
    print(f"deltaA_val shape: {split_payloads['val']['delta_values'].shape}")
    print(f"lambda_test shape: {split_payloads['test']['lambda_values'].shape}")
    print(f"deltaA_test shape: {split_payloads['test']['delta_values'].shape}")


def export_etth1_estimated(args) -> None:
    args.dataset_name = "ETTh1"
    args.split_mode = "ett_hour"
    args.train_end = None
    args.train_ratio = None
    args.date_col = "date"
    args.header_mode = "infer"
    args.sep = ","
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
    real_parser.add_argument("--header-mode", choices=["infer", "none"], default="infer")
    real_parser.add_argument("--sep", default=",")
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
    real_parser.add_argument("--pcmci-max-conds-dim", type=int, default=None)
    real_parser.add_argument("--pcmci-max-conds-py", type=int, default=None)
    real_parser.add_argument("--pcmci-max-conds-px", type=int, default=None)
    real_parser.add_argument("--pcmci-prefilter-mode", choices=["none", "lagcorr_topk"], default="none")
    real_parser.add_argument("--pcmci-prefilter-topk", type=int, default=None)
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
    etth1_parser.add_argument("--pcmci-max-conds-dim", type=int, default=None)
    etth1_parser.add_argument("--pcmci-max-conds-py", type=int, default=None)
    etth1_parser.add_argument("--pcmci-max-conds-px", type=int, default=None)
    etth1_parser.add_argument("--pcmci-prefilter-mode", choices=["none", "lagcorr_topk"], default="none")
    etth1_parser.add_argument("--pcmci-prefilter-topk", type=int, default=None)
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
