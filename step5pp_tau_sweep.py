import os
import json
import csv
import argparse
import subprocess
import sys


def parse_float_list(text):
    out = []
    for p in text.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def run_one(data_dir, cfg_path, out_dir, sanity):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "step5pp_simulate_gated_graph.py",
        "--data_dir",
        data_dir,
        "--config",
        cfg_path,
        "--out_dir",
        out_dir,
    ]
    if sanity:
        cmd.append("--sanity")
    subprocess.check_call(cmd)


def load_json_or_empty(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--modes", type=str, default="hard,soft", help="Comma separated: hard,soft")
    parser.add_argument("--hard_tau_values", type=str, default="0.2,0.4,0.6,0.8")
    parser.add_argument("--soft_w_values", type=str, default="0.0,0.25,0.5,0.75")
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    cfg_path = args.config or os.path.join(args.data_dir, "step5pp_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    out_dir = args.out_dir or os.path.join(args.data_dir, "exports_step5pp")
    os.makedirs(out_dir, exist_ok=True)
    sweep_dir = os.path.join(out_dir, "tau_sweep_runs")
    os.makedirs(sweep_dir, exist_ok=True)

    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    hard_taus = parse_float_list(args.hard_tau_values)
    soft_ws = parse_float_list(args.soft_w_values)

    rows = []
    case_id = 0

    if "hard" in modes:
        for tau in hard_taus:
            case_id += 1
            cfg = dict(base_cfg)
            cfg["gate_mode"] = "hard"
            cfg["tau_hard"] = float(tau)
            case_name = f"{case_id:02d}_hard_tau_{tau:.3f}"
            cfg_case_path = os.path.join(sweep_dir, f"{case_name}.json")
            run_dir = os.path.join(sweep_dir, case_name)
            with open(cfg_case_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            run_one(args.data_dir, cfg_case_path, run_dir, sanity=args.sanity)
            diag = load_json_or_empty(os.path.join(run_dir, "step5pp_diagnostics.json"))
            used = load_json_or_empty(os.path.join(run_dir, "config_used.json"))
            rows.append({
                "case": case_name,
                "gate_mode": "hard",
                "param_name": "tau_hard",
                "param_value": tau,
                "overall_align": diag.get("overall_align"),
                "align_all_pre": diag.get("align_all_pre"),
                "align_all_post": diag.get("align_all_post"),
                "mean_margin_pre": diag.get("mean_margin_pre"),
                "mean_margin_post": diag.get("mean_margin_post"),
                "delta_mask_nnz": diag.get("delta_mask_nnz"),
                "dist_mask_nnz": diag.get("dist_mask_nnz"),
                "check_overall_pass": diag.get("check_overall_pass", used.get("check_overall_pass")),
                "regime_swapped": used.get("regime_swapped"),
                "swap_reason": used.get("swap_reason"),
            })

    if "soft" in modes:
        for w in soft_ws:
            case_id += 1
            cfg = dict(base_cfg)
            cfg["gate_mode"] = "soft"
            cfg["w_soft"] = float(w)
            case_name = f"{case_id:02d}_soft_w_{w:.3f}"
            cfg_case_path = os.path.join(sweep_dir, f"{case_name}.json")
            run_dir = os.path.join(sweep_dir, case_name)
            with open(cfg_case_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            run_one(args.data_dir, cfg_case_path, run_dir, sanity=args.sanity)
            diag = load_json_or_empty(os.path.join(run_dir, "step5pp_diagnostics.json"))
            used = load_json_or_empty(os.path.join(run_dir, "config_used.json"))
            rows.append({
                "case": case_name,
                "gate_mode": "soft",
                "param_name": "w_soft",
                "param_value": w,
                "overall_align": diag.get("overall_align"),
                "align_all_pre": diag.get("align_all_pre"),
                "align_all_post": diag.get("align_all_post"),
                "mean_margin_pre": diag.get("mean_margin_pre"),
                "mean_margin_post": diag.get("mean_margin_post"),
                "delta_mask_nnz": diag.get("delta_mask_nnz"),
                "dist_mask_nnz": diag.get("dist_mask_nnz"),
                "check_overall_pass": diag.get("check_overall_pass", used.get("check_overall_pass")),
                "regime_swapped": used.get("regime_swapped"),
                "swap_reason": used.get("swap_reason"),
            })

    if not rows:
        raise RuntimeError("No sweep cases generated. Check --modes and value lists.")

    out_csv = os.path.join(out_dir, "step5pp_tau_sweep.csv")
    out_md = os.path.join(out_dir, "step5pp_tau_sweep.md")
    headers = list(rows[0].keys())

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join([str(r[h]) for h in headers]) + " |\n")

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")


if __name__ == "__main__":
    main()
