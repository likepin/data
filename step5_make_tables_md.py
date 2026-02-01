import os
import csv
import argparse


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_md(rows, out_path, title, header):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [f"## {title}\n",
             "| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        row_vals = []
        for c in header:
            v = r.get(c, "")
            row_vals.append(str(v))
        lines.append("| " + " | ".join(row_vals) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--tau_csv", type=str, default=None)
    parser.add_argument("--retention_csv", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "exports_step5")
    summary_csv = args.summary_csv or os.path.join(out_dir, "step5_proxy_summary.csv")
    tau_csv = args.tau_csv or os.path.join(out_dir, "step5_tau_sweep.csv")
    retention_csv = args.retention_csv or os.path.join(out_dir, "step5_edge_retention.csv")

    summary_rows = read_csv(summary_csv)
    tau_rows = read_csv(tau_csv)
    retention_rows = read_csv(retention_csv)

    summary_header = [
        "lambda_config", "deltaA_source", "gate_type", "tau", "subset", "subset_q",
        "K_true_change", "TP", "FP", "FN", "Prec", "Rec", "F1", "SHD",
        "SHD_gain_vs_ungated", "F1_delta_vs_ungated"
    ]
    tau_header = [
        "lambda_config", "deltaA_source", "gate_type", "tau",
        "high_SHD_gain", "low_F1_delta", "high_F1", "high_SHD",
        "low_F1", "low_SHD", "pick_best"
    ]
    retention_header = [
        "lambda_config", "deltaA_source", "gate_type", "tau", "subset",
        "K_pred", "TP_change", "FP_change", "retained_ratio",
        "true_retained_ratio", "fp_removed_ratio"
    ]

    write_md(summary_rows, os.path.join(out_dir, "step5_proxy_summary.md"),
             "Table 5-1: Proxy Summary", summary_header)
    write_md(tau_rows, os.path.join(out_dir, "step5_tau_sweep.md"),
             "Table 5-2: Tau Sweep", tau_header)
    write_md(retention_rows, os.path.join(out_dir, "step5_edge_retention.md"),
             "Table 5-3: Edge Retention", retention_header)

    caption_path = os.path.join(out_dir, "caption_templates.md")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write("## Caption Templates\n\n")
        f.write("- Table 5-1: Proxy summary of gating effects across subsets and gate types.\n")
        f.write("- Table 5-2: Tau sweep results; best tau satisfies low_F1_delta >= -0.02.\n")
        f.write("- Table 5-3: Edge retention statistics under gating.\n")
        f.write("- Figure 5-1: Lambda and gated edge counts over time.\n")

    print("=== Step5 make tables md ===")
    print(f"[OK] {os.path.join(out_dir, 'step5_proxy_summary.md')}")
    print(f"[OK] {os.path.join(out_dir, 'step5_tau_sweep.md')}")
    print(f"[OK] {os.path.join(out_dir, 'step5_edge_retention.md')}")
    print(f"[OK] {caption_path}")


if __name__ == "__main__":
    main()
