import os
import json
import argparse
import subprocess
import sys
import traceback
import random

import numpy as np

from step4_export_lambda_variants import export_lambda_variants


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_lambda_file(configs_dir, stem):
    path = os.path.join(configs_dir, stem + ".npz")
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"lambda file not found: {path}")


def write_phaseA_configs(configs_dir, lambda_files):
    base_cfg = {
        "pred_prefix": "cmiknn",
        "gate_mode": "soft",
        "delta_mask_mode": "union_base_predchange",
        "dist_mask_mode": "true_change_only",
        "auto_swap_regimes": False,
        "subset_high_q": 0.9,
        "subset_low_q": 0.5,
    }
    truechange_cfg = dict(base_cfg)
    truechange_cfg["dist_mask_mode"] = "true_change_only"

    cfg_paths = {
        "cfg_phaseA_base": os.path.join(configs_dir, "cfg_phaseA_base.json"),
        "cfg_phaseA_truechange_eval": os.path.join(configs_dir, "cfg_phaseA_truechange_eval.json"),
    }
    write_json(cfg_paths["cfg_phaseA_base"], base_cfg)
    write_json(cfg_paths["cfg_phaseA_truechange_eval"], truechange_cfg)

    run_cfg_map = {
        "score_equal": ("cfg_lambda_equal.json", lambda_files["score_equal"], "score_equal", "main"),
        "score_gating": ("cfg_lambda_gating.json", lambda_files["score_gating"], "score_gating", "main"),
        "score_regime": ("cfg_lambda_regime.json", lambda_files["score_regime"], "score_regime", "main"),
        "lambda_shuffle": ("cfg_lambda_shuffle.json", lambda_files["lambda_shuffle"], "lambda_shuffle", "negative_control"),
        "lambda_constant_05": ("cfg_lambda_const05.json", lambda_files["lambda_constant_05"], "lambda_constant_05", "negative_control"),
        "lambda_constant_10": ("cfg_lambda_const10.json", lambda_files["lambda_constant_10"], "lambda_constant_10", "negative_control"),
    }

    run_defs = []
    for run_name, (cfg_name, lambda_file, lambda_tag, run_type) in run_cfg_map.items():
        cfg_path = os.path.join(configs_dir, cfg_name)
        cfg = dict(base_cfg)
        cfg["lambda_file"] = lambda_file
        cfg["lambda_tag"] = lambda_tag
        cfg["config_name"] = run_name
        cfg["run_type"] = run_type
        write_json(cfg_path, cfg)
        run_defs.append(
            {
                "run_name": run_name,
                "run_type": run_type,
                "lambda_tag": lambda_tag,
                "lambda_file": lambda_file,
                "config_path": cfg_path,
            }
        )
    return run_defs, cfg_paths


def run_one(data_dir, run_def, run_dir):
    cmd = [
        sys.executable,
        "step5pp_simulate_gated_graph.py",
        "--data_dir",
        data_dir,
        "--config",
        run_def["config_path"],
        "--out_dir",
        run_dir,
        "--sanity",
    ]
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--exports_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fail_fast", action="store_true",
                        help="Stop immediately when a run fails. Default is continue-on-error.")
    parser.add_argument("--skip_export_lambdas", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    exports_dir = args.exports_dir or os.path.join(args.data_dir, "exports_step5pp")
    runs_dir = safe_mkdir(os.path.join(exports_dir, "runs"))
    compare_dir = safe_mkdir(os.path.join(exports_dir, "compare"))
    figs_dir = safe_mkdir(os.path.join(exports_dir, "figs"))
    configs_dir = safe_mkdir(os.path.join(exports_dir, "configs"))
    _ = figs_dir  # reserved for follow-up plotting script

    if not args.skip_export_lambdas:
        export_lambda_variants(args.data_dir, exports_dir, seed=int(args.seed))

    lambda_files = {
        "score_equal": find_lambda_file(configs_dir, "lambda_equal"),
        "score_gating": find_lambda_file(configs_dir, "lambda_gating"),
        "score_regime": find_lambda_file(configs_dir, "lambda_regime"),
        "lambda_shuffle": find_lambda_file(configs_dir, "lambda_shuffle"),
        "lambda_constant_05": find_lambda_file(configs_dir, "lambda_const_05"),
        "lambda_constant_10": find_lambda_file(configs_dir, "lambda_const_10"),
    }

    run_defs, cfg_paths = write_phaseA_configs(configs_dir, lambda_files)
    write_json(os.path.join(configs_dir, "cfg_index_phaseA.json"), {"run_defs": run_defs, "cfg_paths": cfg_paths})

    failed_log = os.path.join(compare_dir, "failed_runs.log")
    if os.path.isfile(failed_log):
        os.remove(failed_log)
    results = []
    for run_def in run_defs:
        run_name = run_def["run_name"]
        run_dir = safe_mkdir(os.path.join(runs_dir, run_name))
        source_info = {
            "lambda_strategy": run_name,
            "lambda_tag": run_def["lambda_tag"],
            "lambda_file": run_def["lambda_file"],
            "run_type": run_def["run_type"],
            "seed": int(args.seed),
            "config_path": run_def["config_path"],
        }
        write_json(os.path.join(run_dir, "lambda_source_info.json"), source_info)
        try:
            run_one(args.data_dir, run_def, run_dir)
            status = "ok"
            err_msg = ""
        except Exception:
            status = "failed"
            err_msg = traceback.format_exc()
            with open(failed_log, "a", encoding="utf-8") as f:
                f.write(f"\n=== {run_name} ===\n")
                f.write(err_msg + "\n")
            if args.fail_fast:
                raise
        results.append(
            {
                "run_name": run_name,
                "run_type": run_def["run_type"],
                "status": status,
                "run_dir": run_dir,
                "error": err_msg,
            }
        )

    write_json(os.path.join(compare_dir, "batch_run_status.json"), results)
    print(f"[OK] runs_dir={runs_dir}")
    print(f"[OK] status={os.path.join(compare_dir, 'batch_run_status.json')}")
    if any(r["status"] == "failed" for r in results):
        print(f"[WARN] failed_runs={failed_log}")


if __name__ == "__main__":
    main()
