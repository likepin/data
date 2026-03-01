import os
import csv
import argparse

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception as e:
    plt = None
    HAS_MPL = False
    MPL_ERR = str(e)


def warn(msg):
    print(f"WARN: {msg}")


def to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def to_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "y")


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_figure(fig, out_path, paper_style=False):
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if paper_style:
        root, ext = os.path.splitext(out_path)
        fig.savefig(root + "_paper" + ext, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--exports_dir", type=str, default=None)
    parser.add_argument("--compare_dir", type=str, default=None)
    parser.add_argument("--figs_dir", type=str, default=None)
    parser.add_argument("--paper_style", action="store_true")
    args = parser.parse_args()

    exports_dir = args.exports_dir or os.path.join(args.data_dir, "exports_step5pp")
    compare_dir = args.compare_dir or os.path.join(exports_dir, "compare")
    figs_dir = args.figs_dir or os.path.join(exports_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    if not HAS_MPL:
        warn(f"matplotlib not available: {MPL_ERR}")
        return

    cfg_rows = read_csv(os.path.join(compare_dir, "compare_phaseA_configs.csv"))
    check_rows = read_csv(os.path.join(compare_dir, "compare_phaseA_checks.csv"))
    if not cfg_rows:
        warn("compare_phaseA_configs.csv missing or empty.")
        return

    if args.paper_style:
        plt.rcParams.update({"font.size": 12})

    names = [r.get("lambda_strategy", r.get("config_name", f"run{i}")) for i, r in enumerate(cfg_rows)]
    x = np.arange(len(names))

    # fig 1: align overall
    y_align = np.array([to_float(r.get("align_overall")) for r in cfg_rows], dtype=float)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, y_align, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_title("Align Overall by Strategy")
    ax.set_ylabel("align_overall")
    save_figure(fig, os.path.join(figs_dir, "fig_strategy_bar_align.png"), paper_style=args.paper_style)

    # fig 2: margin pre/post
    y_pre = np.array([to_float(r.get("margin_pre")) for r in cfg_rows], dtype=float)
    y_post = np.array([to_float(r.get("margin_post")) for r in cfg_rows], dtype=float)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    w = 0.35
    ax.bar(x - w / 2, y_pre, width=w, label="margin_pre")
    ax.bar(x + w / 2, y_post, width=w, label="margin_post")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_title("Margin Pre/Post by Strategy")
    ax.legend(loc="best")
    save_figure(fig, os.path.join(figs_dir, "fig_strategy_bar_margin.png"), paper_style=args.paper_style)

    # fig 3: retained gap
    if check_rows:
        gap_map = {r.get("lambda_strategy"): to_float(r.get("retained_gap")) for r in check_rows}
        y_gap = np.array([gap_map.get(n, np.nan) for n in names], dtype=float)
    else:
        warn("compare_phaseA_checks.csv missing, retained gap plot uses NaN.")
        y_gap = np.full_like(y_align, np.nan, dtype=float)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, y_gap, color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_title("Retained Gap by Strategy")
    ax.set_ylabel("retained_low - retained_high")
    save_figure(fig, os.path.join(figs_dir, "fig_strategy_bar_retained_gap.png"), paper_style=args.paper_style)

    # fig 4: dist std triple bar
    y_base = np.array([to_float(r.get("dist_std_base")) for r in cfg_rows], dtype=float)
    y_reg0 = np.array([to_float(r.get("dist_std_reg0")) for r in cfg_rows], dtype=float)
    y_reg1 = np.array([to_float(r.get("dist_std_reg1")) for r in cfg_rows], dtype=float)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    w = 0.25
    ax.bar(x - w, y_base, width=w, label="dist_std_base")
    ax.bar(x, y_reg0, width=w, label="dist_std_reg0")
    ax.bar(x + w, y_reg1, width=w, label="dist_std_reg1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_title("Dist Std by Strategy")
    ax.legend(loc="best")
    save_figure(fig, os.path.join(figs_dir, "fig_strategy_bar_diststd.png"), paper_style=args.paper_style)

    # fig 5: checks heatmap (0/1)
    if check_rows:
        check_map = {r.get("lambda_strategy"): r for r in check_rows}
        keys = ["gate_direction", "high_closer_A0", "low_closer_A1"]
        mat = np.zeros((len(names), len(keys)), dtype=float)
        for i, n in enumerate(names):
            row = check_map.get(n, {})
            for j, k in enumerate(keys):
                mat[i, j] = 1.0 if to_bool(row.get(k)) else 0.0
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(mat, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(keys, rotation=20)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)
        ax.set_title("Core Checks Heatmap")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        save_figure(fig, os.path.join(figs_dir, "fig_strategy_checks_heatmap.png"), paper_style=args.paper_style)
    else:
        warn("compare_phaseA_checks.csv missing; skip checks heatmap.")

    print(f"[OK] {figs_dir}")


if __name__ == "__main__":
    main()
