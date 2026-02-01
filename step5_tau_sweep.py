import os
import csv
import argparse

import numpy as np


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        if v is None or v == "":
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def write_csv(rows, out_path, header):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def write_md(rows, out_path, title, header):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [f"## {title}\n",
             "| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        row_vals = []
        for c in header:
            v = r.get(c, "")
            if isinstance(v, float):
                row_vals.append(f"{v:.6f}")
            else:
                row_vals.append(str(v))
        lines.append("| " + " | ".join(row_vals) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--summary_csv", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "exports_step5")
    summary_csv = args.summary_csv or os.path.join(out_dir, "step5_proxy_summary.csv")

    rows = read_csv(summary_csv)
    if not rows:
        raise RuntimeError("summary CSV is empty.")

    # group by gate_type + tau
    grouped = {}
    for r in rows:
        gate = r.get("gate_type", "")
        tau = r.get("tau", "")
        subset = r.get("subset", "")
        if gate == "ungated":
            continue
        key = (gate, tau)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][subset] = r

    out_rows = []
    for (gate, tau), m in grouped.items():
        high = m.get("high", None)
        low = m.get("low", None)
        if high is None or low is None:
            continue
        row = {
            "lambda_config": high.get("lambda_config", ""),
            "deltaA_source": high.get("deltaA_source", ""),
            "gate_type": gate,
            "tau": to_float(tau),
            "high_SHD_gain": to_float(high.get("SHD_gain_vs_ungated")),
            "low_F1_delta": to_float(low.get("F1_delta_vs_ungated")),
            "high_F1": to_float(high.get("F1")),
            "high_SHD": to_float(high.get("SHD")),
            "low_F1": to_float(low.get("F1")),
            "low_SHD": to_float(low.get("SHD")),
            "pick_best": False,
        }
        out_rows.append(row)

    # pick best: maximize high_SHD_gain with low_F1_delta >= -0.02
    candidates = [r for r in out_rows if np.isfinite(r["high_SHD_gain"]) and np.isfinite(r["low_F1_delta"])
                  and r["low_F1_delta"] >= -0.02]
    if candidates:
        best = max(candidates, key=lambda r: r["high_SHD_gain"])
        for r in out_rows:
            if r is best:
                r["pick_best"] = True

    header = [
        "lambda_config", "deltaA_source", "gate_type", "tau",
        "high_SHD_gain", "low_F1_delta", "high_F1", "high_SHD",
        "low_F1", "low_SHD", "pick_best"
    ]

    out_csv = os.path.join(out_dir, "step5_tau_sweep.csv")
    out_md = os.path.join(out_dir, "step5_tau_sweep.md")
    write_csv(out_rows, out_csv, header)
    write_md(out_rows, out_md, "Table 5-2: Tau Sweep", header)

    print("=== Step5 tau sweep ===")
    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")


if __name__ == "__main__":
    main()
