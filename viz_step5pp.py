import os
import csv
import json
import argparse

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception as e:
    plt = None
    HAS_MPL = False
    MPL_ERR = str(e)

from step5_utils import load_lambda_and_mask


def warn(msg):
    print(f"WARN: {msg}")


_WARNED_ONCE = set()


def warn_once(msg):
    if msg in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(msg)
    warn(msg)


def parse_md_table(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    table_lines = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) < 3:
        return rows
    header = [x.strip() for x in table_lines[0].strip("|").split("|")]
    for ln in table_lines[2:]:
        vals = [x.strip() for x in ln.strip("|").split("|")]
        if len(vals) != len(header):
            continue
        rows.append({header[i]: vals[i] for i in range(len(header))})
    return rows


def read_rows(csv_path, md_path):
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    warn(f"CSV missing: {csv_path}, trying MD: {md_path}")
    return parse_md_table(md_path)


def read_json_or_none(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan


def to_float_or(v, default):
    x = to_float(v)
    try:
        if np.isfinite(float(x)):
            return float(x)
    except Exception:
        pass
    return default


def row_float(row, key, ctx):
    if row is None or key not in row or row.get(key) in ("", None):
        warn_once(f"{ctx}: missing field `{key}`")
        return np.nan
    return to_float(row.get(key))


def selected(only_set, fig_key):
    if not only_set:
        return True
    aliases = {
        "fig1": {"1", "fig1", "lambda_gate"},
        "fig2": {"2", "fig2", "dist_curves"},
        "fig3": {"3", "fig3", "retained"},
        "fig4": {"4", "fig4", "dist_std_bars"},
        "fig5": {"5", "fig5", "high_low_bars"},
        "fig6": {"6", "fig6", "heatmap"},
        "fig7": {"7", "fig7", "align_bars"},
        "fig8": {"8", "fig8", "mask_scatter"},
    }
    return bool(only_set & aliases[fig_key])


def find_run_dir(exports_dir, config_name):
    exact = os.path.join(exports_dir, f"compare_{config_name}")
    if os.path.isdir(exact):
        return exact
    candidates = []
    for name in os.listdir(exports_dir):
        p = os.path.join(exports_dir, name)
        if not os.path.isdir(p):
            continue
        if config_name in name and (
            os.path.isfile(os.path.join(p, "step5pp_summary.csv"))
            or os.path.isfile(os.path.join(p, "step5pp_summary.md"))
        ):
            candidates.append(p)
    if candidates:
        return sorted(candidates)[0]
    if os.path.isfile(os.path.join(exports_dir, "step5pp_summary.csv")):
        return exports_dir
    return None


def load_compare_rows(exports_dir):
    rows = read_rows(
        os.path.join(exports_dir, "compare_configs.csv"),
        os.path.join(exports_dir, "compare_configs.md"),
    )
    if not rows:
        warn("compare_configs.csv/md not found or empty.")
    return rows


def load_summary_rows(run_dir):
    if run_dir is None:
        return []
    return read_rows(
        os.path.join(run_dir, "step5pp_summary.csv"),
        os.path.join(run_dir, "step5pp_summary.md"),
    )


def pick_subset(rows, name):
    for r in rows:
        if r.get("subset") == name:
            return r
    return None


def plot_fig1_lambda_gate(fig_dir, data_dir, gate_mode, tau_hard, t_switch):
    logs = []
    lambda_t, valid_mask, _, t_sw = load_lambda_and_mask(data_dir, logs)
    if t_switch is None:
        t_switch = t_sw
    if gate_mode == "hard":
        g = (lambda_t < tau_hard).astype(np.float32)
    else:
        g = np.clip(1.0 - lambda_t, 0.0, 1.0)
    lam_plot = np.where(valid_mask, lambda_t, np.nan)
    g_plot = np.where(valid_mask, g, np.nan)

    out = os.path.join(fig_dir, "fig1_lambda_gate.png")
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lam_plot, label="lambda(t)", color="tab:blue")
    ax.plot(g_plot, label="g(t)", color="tab:orange")
    if t_switch is not None:
        ax.axvline(int(t_switch), linestyle="--", color="tab:red", label="t_switch")
    ax.set_title("Figure 1: lambda and gate curve")
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_fig2_distance_curves(fig_dir, summary_rows):
    if not summary_rows:
        warn("Figure 2 skipped: summary rows missing.")
        return None
    labels = [r.get("subset", "") for r in summary_rows]
    y_base = [row_float(r, "mean_dist_base", "figure2") for r in summary_rows]
    y_reg0 = [row_float(r, "mean_dist_reg0", "figure2") for r in summary_rows]
    y_reg1 = [row_float(r, "mean_dist_reg1", "figure2") for r in summary_rows]
    x = np.arange(len(labels))
    out = os.path.join(fig_dir, "fig2_distance_curves.png")
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y_base, marker="o", label="dist_base")
    ax.plot(x, y_reg0, marker="o", label="dist_reg0")
    ax.plot(x, y_reg1, marker="o", label="dist_reg1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title("Figure 2: distance curves by subset")
    ax.set_ylabel("mean distance")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_fig3_retained(fig_dir, run_dir):
    if run_dir is None:
        warn("Figure 3 skipped: run dir missing.")
        return None
    path = os.path.join(run_dir, "retained_summary.csv")
    if not os.path.isfile(path):
        warn(f"Figure 3 skipped: missing {path}")
        return None
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            xs.append(int(r["t"]))
            ys.append(float(r["retained_ratio"]))
    out = os.path.join(fig_dir, "fig3_retained_ratio.png")
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, color="tab:purple")
    ax.set_title("Figure 3: retained_ratio(t)")
    ax.set_xlabel("t")
    ax.set_ylabel("retained_ratio")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_fig4_dist_std_bars(fig_dir, compare_rows):
    if not compare_rows:
        warn("Figure 4 skipped: compare rows missing.")
        return None
    names = [r.get("config_name", f"cfg{i}") for i, r in enumerate(compare_rows)]
    base = np.array([row_float(r, "dist_std_base", "figure4") for r in compare_rows], dtype=float)
    reg0 = np.array([row_float(r, "dist_std_reg0", "figure4") for r in compare_rows], dtype=float)
    reg1 = np.array([row_float(r, "dist_std_reg1", "figure4") for r in compare_rows], dtype=float)
    x = np.arange(len(names))
    w = 0.25
    out = os.path.join(fig_dir, "fig4_dist_std_bars.png")
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x - w, base, width=w, label="dist_std_base")
    ax.bar(x, reg0, width=w, label="dist_std_reg0")
    ax.bar(x + w, reg1, width=w, label="dist_std_reg1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_title("Figure 4: dist std by config")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_fig5_high_low_grouped(fig_dir, compare_rows):
    if not compare_rows:
        warn("Figure 5 skipped: compare rows missing.")
        return None
    names = [r.get("config_name", f"cfg{i}") for i, r in enumerate(compare_rows)]
    metrics = [
        ("mean_lambda", "high_mean_lambda", "low_mean_lambda"),
        ("mean_gate_weight", "high_mean_gate_weight", "low_mean_gate_weight"),
        ("mean_dist_base", "high_mean_dist_base", "low_mean_dist_base"),
        ("mean_retained_ratio", "high_mean_retained_ratio", "low_mean_retained_ratio"),
    ]
    out = os.path.join(fig_dir, "fig5_high_low_grouped.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.reshape(-1)
    x = np.arange(len(names))
    w = 0.35
    for i, (title, h_key, l_key) in enumerate(metrics):
        high = np.array([row_float(r, h_key, "figure5") for r in compare_rows], dtype=float)
        low = np.array([row_float(r, l_key, "figure5") for r in compare_rows], dtype=float)
        ax = axes[i]
        ax.bar(x - w / 2, high, width=w, label="high_non_sat")
        ax.bar(x + w / 2, low, width=w, label="low")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.set_title(title)
        if i == 0:
            ax.legend(loc="best")
    fig.suptitle("Figure 5: high_non_sat vs low grouped metrics")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_fig6_heatmap(fig_dir, compare_rows):
    if not compare_rows:
        warn("Figure 6 skipped: compare rows missing.")
        return None
    names = [r.get("config_name", f"cfg{i}") for i, r in enumerate(compare_rows)]
    keys = [
        "dist_std_base",
        "dist_std_reg0",
        "dist_std_reg1",
        "align_pre",
        "align_post",
        "align_overall",
        "high_mean_lambda",
        "low_mean_lambda",
        "high_mean_gate_weight",
        "low_mean_gate_weight",
    ]
    mat = np.array([[row_float(r, k, "figure6") for k in keys] for r in compare_rows], dtype=float)
    if not np.isfinite(mat).any():
        warn("Figure 6 skipped: all heatmap values are NaN.")
        return None
    out = os.path.join(fig_dir, "fig6_config_heatmap.png")
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_title("Figure 6: config metric heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_fig7_align_bars(fig_dir, compare_rows):
    if not compare_rows:
        warn("Figure 7 skipped: compare rows missing.")
        return None
    names = [r.get("config_name", f"cfg{i}") for i, r in enumerate(compare_rows)]
    pre = np.array([row_float(r, "align_pre", "figure7") for r in compare_rows], dtype=float)
    post = np.array([row_float(r, "align_post", "figure7") for r in compare_rows], dtype=float)
    x = np.arange(len(names))
    w = 0.35
    out = os.path.join(fig_dir, "fig7_align_bars.png")
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x - w / 2, pre, width=w, label="align_pre")
    ax.bar(x + w / 2, post, width=w, label="align_post")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_title("Figure 7: align_pre vs align_post")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_fig8_scatter(fig_dir, compare_rows):
    if not compare_rows:
        warn("Figure 8 skipped: compare rows missing.")
        return None
    names = [r.get("config_name", f"cfg{i}") for i, r in enumerate(compare_rows)]
    xs = np.array([row_float(r, "dist_mask_nnz", "figure8") for r in compare_rows], dtype=float)
    ys = np.array([row_float(r, "dist_std_reg0", "figure8") for r in compare_rows], dtype=float)
    out = os.path.join(fig_dir, "fig8_masknnz_vs_diststd_scatter.png")
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xs, ys, s=60, color="tab:blue")
    for i, n in enumerate(names):
        if np.isfinite(xs[i]) and np.isfinite(ys[i]):
            ax.text(xs[i], ys[i], n, fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("dist_mask_nnz")
    ax.set_ylabel("dist_std_reg0")
    ax.set_title("Figure 8: dist_mask_nnz vs dist_std_reg0")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def write_fig_index(exports_dir, entries):
    path = os.path.join(exports_dir, "fig_index.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("| file | purpose |\n")
        f.write("| --- | --- |\n")
        for fp, desc in entries:
            rel = os.path.relpath(fp, exports_dir)
            f.write(f"| {rel} | {desc} |\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--exports_dir", type=str, default=None)
    parser.add_argument("--only", type=str, default="", help="Comma list: 1,2,... or fig1,fig2,...")
    parser.add_argument("--ref_config", type=str, default=None, help="Use this config_name as reference for fig2/fig3.")
    args = parser.parse_args()

    exports_dir = args.exports_dir or os.path.join(args.data_dir, "exports_step5pp")
    fig_dir = os.path.join(exports_dir, "figs_step5pp")
    os.makedirs(fig_dir, exist_ok=True)

    only_set = {x.strip().lower() for x in args.only.split(",") if x.strip()}
    compare_rows = load_compare_rows(exports_dir)
    if not compare_rows:
        warn("No compare rows available; some figures will be skipped.")

    ref_cfg = args.ref_config or (compare_rows[0].get("config_name") if compare_rows else None)
    ref_run_dir = find_run_dir(exports_dir, ref_cfg) if ref_cfg else None
    if ref_cfg and ref_run_dir is None:
        warn(f"Cannot find run dir for reference config {ref_cfg}.")

    ref_summary = load_summary_rows(ref_run_dir)
    ref_used = read_json_or_none(os.path.join(ref_run_dir, "config_used.json")) if ref_run_dir else None
    if ref_used is None:
        ref_used = read_json_or_none(os.path.join(exports_dir, "config_used.json"))
    gate_mode = (ref_used or {}).get("gate_mode", "soft")
    tau_hard = to_float_or((ref_used or {}).get("tau_hard"), 0.8)

    meta = read_json_or_none(os.path.join(args.data_dir, "meta.json")) or {}
    t_switch = meta.get("t_switch")

    entries = []

    if not HAS_MPL:
        warn(f"matplotlib not available: {MPL_ERR}")
        warn("No figures generated. Install matplotlib to enable plotting.")
        idx = write_fig_index(exports_dir, [])
        print(f"[OK] {idx}")
        return

    if selected(only_set, "fig1"):
        try:
            p = plot_fig1_lambda_gate(fig_dir, args.data_dir, gate_mode, tau_hard, t_switch)
            entries.append((p, "lambda(t) and gate g(t) with t_switch"))
        except Exception as e:
            warn(f"Figure 1 failed: {e}")

    if selected(only_set, "fig2"):
        p = plot_fig2_distance_curves(fig_dir, ref_summary)
        if p:
            entries.append((p, "distance curves (dist_base/reg0/reg1) from subset summary"))

    if selected(only_set, "fig3"):
        p = plot_fig3_retained(fig_dir, ref_run_dir)
        if p:
            entries.append((p, "retained_ratio(t)"))

    if selected(only_set, "fig4"):
        p = plot_fig4_dist_std_bars(fig_dir, compare_rows)
        if p:
            entries.append((p, "bar chart of dist_std_base/reg0/reg1 across configs"))

    if selected(only_set, "fig5"):
        p = plot_fig5_high_low_grouped(fig_dir, compare_rows)
        if p:
            entries.append((p, "grouped bars: high_non_sat vs low metrics"))

    if selected(only_set, "fig6"):
        p = plot_fig6_heatmap(fig_dir, compare_rows)
        if p:
            entries.append((p, "heatmap of key metrics by config"))

    if selected(only_set, "fig7"):
        p = plot_fig7_align_bars(fig_dir, compare_rows)
        if p:
            entries.append((p, "align_pre and align_post bars"))

    if selected(only_set, "fig8"):
        p = plot_fig8_scatter(fig_dir, compare_rows)
        if p:
            entries.append((p, "scatter: dist_mask_nnz vs dist_std_reg0 with config labels"))

    idx = write_fig_index(exports_dir, entries)
    print(f"[OK] {idx}")
    for fp, _ in entries:
        print(f"[OK] {fp}")


if __name__ == "__main__":
    main()
