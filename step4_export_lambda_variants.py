import os
import csv
import json
import argparse
import random

import numpy as np

from step5pp_utils import compute_lambda_kmeans, pick_lambda_configs_from_step4


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_lambda_variant(path_base, lambda_t, valid_mask):
    np.save(path_base + ".npy", lambda_t.astype(np.float64))
    np.savez(path_base + ".npz", lambda_t=lambda_t.astype(np.float64), valid_mask=valid_mask.astype(bool))


def block_shuffle(values, block_size, seed):
    arr = np.array(values, dtype=np.float64).copy()
    if arr.size == 0:
        return arr
    block_size = max(1, int(block_size))
    blocks = [arr[i:i + block_size] for i in range(0, arr.size, block_size)]
    rng = np.random.RandomState(seed)
    rng.shuffle(blocks)
    return np.concatenate(blocks, axis=0)


def export_lambda_variants(data_dir, exports_dir, seed=2026):
    random.seed(seed)
    np.random.seed(seed)

    x_path = os.path.join(data_dir, "X.npy")
    if not os.path.isfile(x_path):
        raise FileNotFoundError(f"X.npy not found: {x_path}")
    X = np.load(x_path)

    configs_dir = safe_mkdir(os.path.join(exports_dir, "configs"))
    rows = []
    main_variants = {}

    strategy_map = [
        ("score_equal", "equal"),
        ("score_gating", "gating"),
        ("score_regime", "regime"),
    ]

    for score_key, short_name in strategy_map:
        cfgs = pick_lambda_configs_from_step4(data_dir, score_key, top_m=1)
        if not cfgs:
            raise RuntimeError(f"No lambda config found for {score_key}")
        c = cfgs[0]
        lambda_t, valid_mask = compute_lambda_kmeans(X, c["window"], c["k"], seed=seed)
        base = os.path.join(configs_dir, f"lambda_{short_name}")
        save_lambda_variant(base, lambda_t, valid_mask)
        rows.append(
            {
                "lambda_strategy": f"score_{short_name}",
                "lambda_file_npy": base + ".npy",
                "lambda_file_npz": base + ".npz",
                "source_type": "step4_rescored",
                "run_type": "main",
                "control_family": "main",
                "source_score_key": score_key,
                "window": c["window"],
                "k": c["k"],
                "score": c["score"],
                "seed": seed,
            }
        )
        main_variants[short_name] = {"lambda_t": lambda_t, "valid_mask": valid_mask}

    # Controls based on gating if available, fallback to equal/regime.
    base_key = "gating" if "gating" in main_variants else ("equal" if "equal" in main_variants else "regime")
    base_lambda = np.array(main_variants[base_key]["lambda_t"], dtype=np.float64)
    base_valid = np.array(main_variants[base_key]["valid_mask"], dtype=bool)

    # shuffle(global): keep same valid positions and value distribution.
    shuffled = base_lambda.copy()
    valid_vals = shuffled[base_valid].copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_vals)
    shuffled[base_valid] = valid_vals
    shuffle_base = os.path.join(configs_dir, "lambda_shuffle_global")
    save_lambda_variant(shuffle_base, shuffled, base_valid)
    rows.append(
        {
            "lambda_strategy": "lambda_shuffle_global",
            "lambda_file_npy": shuffle_base + ".npy",
            "lambda_file_npz": shuffle_base + ".npz",
            "source_type": f"shuffle_from_{base_key}",
            "run_type": "negative_control",
            "control_family": "shuffle_global",
            "source_score_key": "",
            "window": "",
            "k": "",
            "score": "",
            "seed": seed,
        }
    )
    # Backward-compatible alias.
    save_lambda_variant(os.path.join(configs_dir, "lambda_shuffle"), shuffled, base_valid)

    # block shuffle family: preserve local continuity in blocks but break global alignment.
    block_sizes = (50, 100, 200, 500)
    for i, block_size in enumerate(block_sizes):
        block_vals = block_shuffle(base_lambda[base_valid], block_size=block_size, seed=seed + 17 + i)
        lambda_block = np.full_like(base_lambda, np.nan, dtype=np.float64)
        lambda_block[base_valid] = block_vals
        block_base = os.path.join(configs_dir, f"lambda_block_shuffle_{block_size}")
        save_lambda_variant(block_base, lambda_block, base_valid)
        rows.append(
            {
                "lambda_strategy": f"lambda_block_shuffle_{block_size}",
                "lambda_file_npy": block_base + ".npy",
                "lambda_file_npz": block_base + ".npz",
                "source_type": f"block_shuffle_{block_size}_from_{base_key}",
                "run_type": "negative_control",
                "control_family": "block_shuffle",
                "source_score_key": "",
                "window": "",
                "k": "",
                "score": "",
                "seed": seed,
            }
        )
    # Backward-compatible alias.
    save_lambda_variant(os.path.join(configs_dir, "lambda_block_shuffle"), lambda_block, base_valid)

    # shifts: keep value distribution and local smoothness, but wrong temporal alignment.
    for shift in (100, 300, 600, 1000):
        vals = base_lambda[base_valid]
        rolled = np.roll(vals, int(shift))
        arr = np.full_like(base_lambda, np.nan, dtype=np.float64)
        arr[base_valid] = rolled
        out_base = os.path.join(configs_dir, f"lambda_shift_{shift}")
        save_lambda_variant(out_base, arr, base_valid)
        rows.append(
            {
                "lambda_strategy": f"lambda_shift_{shift}",
                "lambda_file_npy": out_base + ".npy",
                "lambda_file_npz": out_base + ".npz",
                "source_type": f"shift_{shift}_from_{base_key}",
                "run_type": "negative_control",
                "control_family": "shift",
                "source_score_key": "",
                "window": "",
                "k": "",
                "score": "",
                "seed": seed,
            }
        )

    # constants
    for val, tag in [(0.5, "constant_05"), (1.0, "constant_10")]:
        arr = np.full_like(base_lambda, np.nan, dtype=np.float64)
        arr[base_valid] = float(val)
        out_base = os.path.join(configs_dir, f"lambda_{tag}")
        save_lambda_variant(out_base, arr, base_valid)
        rows.append(
            {
                "lambda_strategy": f"lambda_{tag}",
                "lambda_file_npy": out_base + ".npy",
                "lambda_file_npz": out_base + ".npz",
                "source_type": "constant",
                "run_type": "negative_control",
                "control_family": "constant",
                "source_score_key": "",
                "window": "",
                "k": "",
                "score": "",
                "seed": seed,
            }
        )
    save_lambda_variant(os.path.join(configs_dir, "lambda_const_05"), np.where(base_valid, 0.5, np.nan), base_valid)
    save_lambda_variant(os.path.join(configs_dir, "lambda_const_10"), np.where(base_valid, 1.0, np.nan), base_valid)

    meta_csv = os.path.join(configs_dir, "lambda_metadata.csv")
    headers = [
        "lambda_strategy",
        "lambda_file_npy",
        "lambda_file_npz",
        "source_type",
        "run_type",
        "control_family",
        "source_score_key",
        "window",
        "k",
        "score",
        "seed",
    ]
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    meta_json = os.path.join(configs_dir, "lambda_metadata.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"[OK] {meta_csv}")
    print(f"[OK] {meta_json}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--exports_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    exports_dir = args.exports_dir or os.path.join(args.data_dir, "exports_step5pp")
    safe_mkdir(exports_dir)
    export_lambda_variants(args.data_dir, exports_dir, seed=int(args.seed))


if __name__ == "__main__":
    main()
