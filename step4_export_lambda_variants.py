import os
import csv
import json
import argparse
import random
import hashlib
import itertools

import numpy as np

from step5pp_utils import compute_lambda_kmeans, pick_lambda_configs_from_step4


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_lambda_variant(path_base, lambda_t, valid_mask):
    np.save(path_base + ".npy", lambda_t.astype(np.float64))
    np.savez(path_base + ".npz", lambda_t=lambda_t.astype(np.float64), valid_mask=valid_mask.astype(bool))


def lambda_hash_round6(lambda_t, valid_mask):
    arr = np.asarray(lambda_t, dtype=np.float64).reshape(-1)
    m = np.asarray(valid_mask, dtype=bool).reshape(-1) & np.isfinite(arr)
    if int(m.sum()) == 0:
        return ""
    payload = np.round(arr[m], 6).astype(np.float64).tobytes()
    return hashlib.sha1(payload).hexdigest()


def lambda_stats(lambda_t, valid_mask):
    arr = np.asarray(lambda_t, dtype=np.float64).reshape(-1)
    m = np.asarray(valid_mask, dtype=bool).reshape(-1) & np.isfinite(arr)
    if int(m.sum()) == 0:
        return {
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "lambda_min": np.nan,
            "lambda_max": np.nan,
            "valid_count": 0,
            "valid_ratio": 0.0,
        }
    v = arr[m]
    return {
        "lambda_mean": float(v.mean()),
        "lambda_std": float(v.std()),
        "lambda_min": float(v.min()),
        "lambda_max": float(v.max()),
        "valid_count": int(v.size),
        "valid_ratio": float(v.size / float(arr.size)) if arr.size > 0 else 0.0,
    }


def pairwise_corr_and_mad(a, a_valid, b, b_valid):
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    ma = np.asarray(a_valid, dtype=bool).reshape(-1)
    mb = np.asarray(b_valid, dtype=bool).reshape(-1)
    m = ma & mb & np.isfinite(aa) & np.isfinite(bb)
    if int(m.sum()) < 2:
        return np.nan, np.nan, int(m.sum())
    x = aa[m]
    y = bb[m]
    sx = float(x.std())
    sy = float(y.std())
    if sx <= 1e-12 or sy <= 1e-12:
        corr = np.nan
    else:
        corr = float(np.corrcoef(x, y)[0, 1])
    mad = float(np.abs(x - y).mean())
    return corr, mad, int(m.sum())


def is_collapse_pair(a, b, corr_thr=0.99, mad_thr=1e-3):
    corr, mad, overlap = pairwise_corr_and_mad(
        a["lambda_t"], a["valid_mask"], b["lambda_t"], b["valid_mask"]
    )
    hash_same = bool(a.get("lambda_hash_round6", "") and b.get("lambda_hash_round6", "") and a["lambda_hash_round6"] == b["lambda_hash_round6"])
    collapse = bool(
        hash_same or
        (np.isfinite(corr) and corr > float(corr_thr)) or
        (np.isfinite(mad) and mad < float(mad_thr))
    )
    return collapse, {
        "corr": corr,
        "mean_abs_diff": mad,
        "overlap_count": int(overlap),
        "hash_same": hash_same,
    }


def extract_metric_subset(raw_row, keep_norm):
    out = {}
    for k, v in dict(raw_row or {}).items():
        if keep_norm:
            if k.startswith("n_") or k.startswith("score_norm_"):
                out[k] = v
        else:
            if k.startswith("n_") or k.startswith("score_norm_"):
                continue
            out[k] = v
    return out


def block_shuffle(values, block_size, seed):
    arr = np.array(values, dtype=np.float64).copy()
    if arr.size == 0:
        return arr
    block_size = max(1, int(block_size))
    blocks = [arr[i:i + block_size] for i in range(0, arr.size, block_size)]
    rng = np.random.RandomState(seed)
    rng.shuffle(blocks)
    return np.concatenate(blocks, axis=0)


def export_lambda_variants(data_dir, exports_dir, seed=2026, top_m=5, corr_thr=0.99, mad_thr=1e-3):
    random.seed(seed)
    np.random.seed(seed)

    x_path = os.path.join(data_dir, "X.npy")
    if not os.path.isfile(x_path):
        raise FileNotFoundError(f"X.npy not found: {x_path}")
    X = np.load(x_path)
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            data_meta = json.load(f)
    else:
        data_meta = {}
    default_t_switch = data_meta.get("t_switch")
    default_switch_window = data_meta.get("switch_window", 200)
    default_band = data_meta.get("band", default_switch_window)

    configs_dir = safe_mkdir(os.path.join(exports_dir, "configs"))
    rows = []
    main_variants = {}
    variant_summary_rows = []
    pairwise_rows = []

    strategy_map = [
        ("score_equal", "equal"),
        ("score_gating", "gating"),
        ("score_regime", "regime"),
    ]

    candidate_map = {}
    for score_key, short_name in strategy_map:
        cfgs = pick_lambda_configs_from_step4(data_dir, score_key, top_m=int(top_m))
        if not cfgs:
            raise RuntimeError(f"No lambda config found for {score_key}")
        cands = []
        for rank_idx, c in enumerate(cfgs, start=1):
            lambda_t, valid_mask = compute_lambda_kmeans(X, c["window"], c["k"], seed=seed)
            lam_hash = lambda_hash_round6(lambda_t, valid_mask)
            cands.append({
                "score_key": score_key,
                "short_name": short_name,
                "rank": int(rank_idx),
                "window": int(c["window"]),
                "k": int(c["k"]),
                "score": float(c["score"]),
                "source_csv": c.get("source_csv", ""),
                "picked_row_raw": dict(c.get("picked_row_raw", {})),
                "lambda_t": lambda_t,
                "valid_mask": valid_mask,
                "lambda_hash_round6": lam_hash,
            })
        candidate_map[short_name] = cands

    chosen = {}
    chosen_list = []
    for score_key, short_name in strategy_map:
        cands = candidate_map[short_name]
        selected = None
        selected_reason = ""
        selected_collapses = []
        for cand in cands:
            collisions = []
            for prev in chosen_list:
                is_col, diag = is_collapse_pair(cand, prev, corr_thr=corr_thr, mad_thr=mad_thr)
                if is_col:
                    collisions.append({
                        "against": prev["variant_name"],
                        "corr": diag["corr"],
                        "mean_abs_diff": diag["mean_abs_diff"],
                        "hash_same": diag["hash_same"],
                        "overlap_count": diag["overlap_count"],
                    })
            if not collisions:
                selected = cand
                selected_reason = (
                    f"top{cand['rank']} by {score_key}; non-collapse "
                    f"(corr<= {corr_thr}, mad>= {mad_thr})"
                )
                selected_collapses = []
                break
            selected_collapses = collisions
        if selected is None:
            selected = cands[0]
            selected_reason = f"fallback_top1_by_{score_key}_collapse_unavoidable"
        selected["variant_name"] = f"score_{short_name}"
        selected["picked_reason"] = selected_reason
        selected["collapse_with_existing"] = bool(len(selected_collapses) > 0)
        selected["collapse_with_existing_details"] = selected_collapses
        chosen[short_name] = selected
        chosen_list.append(selected)

    for score_key, short_name in strategy_map:
        c = chosen[short_name]
        picked_raw = c.get("picked_row_raw", {})
        lambda_t = c["lambda_t"]
        valid_mask = c["valid_mask"]
        base = os.path.join(configs_dir, f"lambda_{short_name}")
        save_lambda_variant(base, lambda_t, valid_mask)
        lam_hash = c["lambda_hash_round6"]
        lam_stats = lambda_stats(lambda_t, valid_mask)
        rows.append(
            {
                "lambda_strategy": f"score_{short_name}",
                "lambda_file_npy": base + ".npy",
                "lambda_file_npz": base + ".npz",
                "source_type": "step4_rescored",
                "run_type": "main",
                "control_family": "main",
                "source_score_key": c.get("score_key", score_key),
                "window": c["window"],
                "k": c["k"],
                "score": c["score"],
                "seed": seed,
            }
        )
        main_variants[short_name] = {
            "lambda_t": lambda_t,
            "valid_mask": valid_mask,
            "lambda_hash_round6": lam_hash,
            "window": c["window"],
            "k": c["k"],
            "score": c["score"],
            "variant_name": c["variant_name"],
        }

        variant_summary_rows.append(
            {
                "variant_name": f"score_{short_name}",
                "lambda_path": base + ".npy",
                "lambda_hash_round6": lam_hash,
                "lambda_mean": lam_stats["lambda_mean"],
                "lambda_std": lam_stats["lambda_std"],
                "lambda_min": lam_stats["lambda_min"],
                "lambda_max": lam_stats["lambda_max"],
                "valid_count": lam_stats["valid_count"],
                "valid_ratio": lam_stats["valid_ratio"],
                "window": c["window"],
                "k": c["k"],
                "score_key": c.get("score_key", score_key),
                "score": c["score"],
                "source_csv": c.get("source_csv", ""),
                "selected_rank": c.get("rank"),
                "picked_reason": c.get("picked_reason", ""),
                "collapse_with_existing": bool(c.get("collapse_with_existing", False)),
            }
        )

        picked_reason = c.get("picked_reason", "")
        meta_obj = {
            "variant_name": f"score_{short_name}",
            "lambda_file_npy": base + ".npy",
            "lambda_file_npz": base + ".npz",
            "lambda_hash_round6": lam_hash,
            "lambda_stats": lam_stats,
            "picked_row": {
                "window": c["window"],
                "k": c["k"],
                "score_key": c.get("score_key", score_key),
                "score": c["score"],
                "source_csv": c.get("source_csv", ""),
                "selected_rank": c.get("rank"),
            },
            "picked_metrics_raw": extract_metric_subset(picked_raw, keep_norm=False),
            "picked_metrics_norm": extract_metric_subset(picked_raw, keep_norm=True),
            "picked_reason": picked_reason,
            "collapse_with_existing": bool(c.get("collapse_with_existing", False)),
            "collapse_with_existing_details": c.get("collapse_with_existing_details", []),
            "collapse_rule": {
                "corr_thr": float(corr_thr),
                "mad_thr": float(mad_thr),
            },
            "t_switch": picked_raw.get("t_switch", default_t_switch),
            "switch_window": picked_raw.get("switch_window", default_switch_window),
            "band": picked_raw.get("band", default_band),
            "seed": int(seed),
        }
        meta_path = os.path.join(configs_dir, f"lambda_variant_meta_score_{short_name}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_obj, f, indent=2)

    # Pairwise diagnostics among main strategies.
    for a_name, b_name in itertools.combinations(sorted(main_variants.keys()), 2):
        a = main_variants[a_name]
        b = main_variants[b_name]
        corr, mad, overlap = pairwise_corr_and_mad(
            a["lambda_t"], a["valid_mask"], b["lambda_t"], b["valid_mask"]
        )
        pairwise_rows.append(
            {
                "variant_a": f"score_{a_name}",
                "variant_b": f"score_{b_name}",
                "corr": corr,
                "mean_abs_diff": mad,
                "overlap_count": overlap,
                "hash_same": bool(a.get("lambda_hash_round6", "") == b.get("lambda_hash_round6", "")),
            }
        )

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

    # Summary artifacts for collapse diagnostics.
    def write_rows(path, items):
        if not items:
            return
        keys = sorted(set().union(*[set(x.keys()) for x in items]))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for it in items:
                w.writerow(it)

    variants_csv = os.path.join(exports_dir, "lambda_variants_summary.csv")
    variants_json = os.path.join(exports_dir, "lambda_variants_summary.json")
    pairwise_csv = os.path.join(exports_dir, "lambda_pairwise.csv")
    pairwise_json = os.path.join(exports_dir, "lambda_pairwise.json")
    write_rows(variants_csv, variant_summary_rows)
    write_rows(pairwise_csv, pairwise_rows)
    with open(variants_json, "w", encoding="utf-8") as f:
        json.dump(variant_summary_rows, f, indent=2)
    with open(pairwise_json, "w", encoding="utf-8") as f:
        json.dump(pairwise_rows, f, indent=2)

    print(f"[OK] {meta_csv}")
    print(f"[OK] {meta_json}")
    print(f"[OK] {variants_csv}")
    print(f"[OK] {variants_json}")
    print(f"[OK] {pairwise_csv}")
    print(f"[OK] {pairwise_json}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--exports_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top_m", type=int, default=5)
    parser.add_argument("--corr_thr", type=float, default=0.99)
    parser.add_argument("--mad_thr", type=float, default=1e-3)
    args = parser.parse_args()

    exports_dir = args.exports_dir or os.path.join(args.data_dir, "exports_step5pp")
    safe_mkdir(exports_dir)
    export_lambda_variants(
        args.data_dir,
        exports_dir,
        seed=int(args.seed),
        top_m=int(args.top_m),
        corr_thr=float(args.corr_thr),
        mad_thr=float(args.mad_thr),
    )


if __name__ == "__main__":
    main()
