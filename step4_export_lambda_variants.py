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

    # shuffle: keep same valid positions and value distribution.
    shuffled = base_lambda.copy()
    valid_vals = shuffled[base_valid].copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_vals)
    shuffled[base_valid] = valid_vals
    shuffle_base = os.path.join(configs_dir, "lambda_shuffle")
    save_lambda_variant(shuffle_base, shuffled, base_valid)
    rows.append(
        {
            "lambda_strategy": "lambda_shuffle",
            "lambda_file_npy": shuffle_base + ".npy",
            "lambda_file_npz": shuffle_base + ".npz",
            "source_type": f"shuffle_from_{base_key}",
            "source_score_key": "",
            "window": "",
            "k": "",
            "score": "",
            "seed": seed,
        }
    )

    # constants
    for val, tag in [(0.5, "const_05"), (1.0, "const_10")]:
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
                "source_score_key": "",
                "window": "",
                "k": "",
                "score": "",
                "seed": seed,
            }
        )

    meta_csv = os.path.join(configs_dir, "lambda_metadata.csv")
    headers = [
        "lambda_strategy",
        "lambda_file_npy",
        "lambda_file_npz",
        "source_type",
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
