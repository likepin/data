import os
import numpy as np
import csv
from datetime import datetime

# plotting (optional but recommended)
import matplotlib.pyplot as plt


# -------------------------
# helpers
# -------------------------



def write_summary_markdown(summary_rows, out_path, title="Summary (Change Detection)"):
    safe_mkdir(os.path.dirname(out_path))
    header = ["prefix","mask","score_type","TP","FP","FN","Prec","Rec","F1","SHD","K_true_change"]
    lines = [f"## {title}\n",
             "| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"]*len(header)) + " |"]
    for r in summary_rows:
        lines.append("| " + " | ".join([
            str(r.get("prefix","")),
            str(r.get("mask","")),
            str(r.get("score_type","")),
            str(r.get("TP","")),
            str(r.get("FP","")),
            str(r.get("FN","")),
            f"{float(r.get('Prec',0)):.3f}",
            f"{float(r.get('Rec',0)):.3f}",
            f"{float(r.get('F1',0)):.3f}",
            str(r.get("SHD","")),
            str(r.get("K_true_change","")),
        ]) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def write_summary_latex(summary_rows, out_path, caption="Summary (Change Detection)", label="tab:summary_change"):
    safe_mkdir(os.path.dirname(out_path))
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l l l r r r r r r r}")
    lines.append(r"\hline")
    lines.append(r"Method & Mask & Score & TP & FP & FN & Prec & Rec & F1 & SHD \\")
    lines.append(r"\hline")
    for r in summary_rows:
        lines.append(
            f"{r.get('prefix','')} & {r.get('mask','')} & {r.get('score_type','')} & "
            f"{int(r.get('TP',0))} & {int(r.get('FP',0))} & {int(r.get('FN',0))} & "
            f"{float(r.get('Prec',0)):.3f} & {float(r.get('Rec',0)):.3f} & {float(r.get('F1',0)):.3f} & "
            f"{int(r.get('SHD',0))} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def best_signed_val_over_lags(val_matrix):
    # val_matrix[src, tgt, lag]
    vals = val_matrix[:, :, 1:]
    idx = np.argmax(np.abs(vals), axis=2)  # (src,tgt) in 0..tau_max-1
    out = np.zeros(vals.shape[:2], dtype=np.float32)
    lag_out = np.zeros(vals.shape[:2], dtype=np.int32)
    for src in range(out.shape[0]):
        for tgt in range(out.shape[1]):
            k = int(idx[src, tgt])
            out[src, tgt] = vals[src, tgt, k]
            lag_out[src, tgt] = k + 1  # 1..tau_max
    return out, lag_out

def metrics_excluding_diag(adj_true, adj_hat):
    N = adj_true.shape[0]
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    y_true = adj_true[mask].astype(int)
    y_hat  = adj_hat[mask].astype(int)
    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    shd  = int(np.abs(y_true - y_hat).sum())
    return tp, fp, fn, prec, rec, f1, shd

def build_candidate_masks(adj_base_tgt_src, a0_tgt_src, a1_tgt_src):
    cand_true = (adj_base_tgt_src.T == 1)                   # (src,tgt)
    cand_pred = ((a0_tgt_src == 1) & (a1_tgt_src == 1)).T    # (src,tgt)
    N = adj_base_tgt_src.shape[0]
    for k in range(N):
        cand_true[k, k] = False
        cand_pred[k, k] = False
    return cand_true, cand_pred

def make_scores(s0, s1):
    score_mag  = np.abs(np.abs(s1) - np.abs(s0))
    sign_flip  = (np.sign(s0) != np.sign(s1)).astype(np.float32)
    score_flip = sign_flip * (np.abs(s0) + np.abs(s1))
    return score_mag, score_flip

def topk_edges(score, candidate_mask, K):
    score2 = score.copy()
    score2[~candidate_mask] = -np.inf
    flat = np.argsort(score2.reshape(-1))[::-1]
    N = score.shape[0]
    out = []
    for idx in flat:
        src = idx // N
        tgt = idx % N
        if not candidate_mask[src, tgt]:
            continue
        sc = float(score2[src, tgt])
        if not np.isfinite(sc):
            continue
        out.append((int(src), int(tgt), sc))
        if len(out) >= K:
            break
    return out

def edges_to_pred_adj(edges, N):
    pred = np.zeros((N, N), dtype=np.int32)  # (tgt,src)
    for src, tgt, _ in edges:
        pred[tgt, src] = 1
    return pred

def export_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    header = sorted(keys)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


# -------------------------
# truth extraction
# -------------------------
def build_truth_change_table(data_dir):
    A_base  = np.load(os.path.join(data_dir, "A_base.npy"))    # (L,tgt,src)
    DeltaA  = np.load(os.path.join(data_dir, "DeltaA.npy"))    # (L,tgt,src)
    adj_chg = np.load(os.path.join(data_dir, "adj_change.npy"))# (tgt,src)
    L, N, _ = DeltaA.shape

    truth_rows = []
    truth_map = {}

    for tgt in range(N):
        for src in range(N):
            if src == tgt:
                continue
            if adj_chg[tgt, src] != 1:
                continue

            lags = []
            for l in range(L):
                d = float(DeltaA[l, tgt, src])
                if abs(d) > 1e-8:
                    base_w = float(A_base[l, tgt, src])
                    new_w  = float(A_base[l, tgt, src] + DeltaA[l, tgt, src])
                    lags.append((l + 1, base_w, d, new_w))

            if len(lags) == 0:
                continue

            lag_star, base_w_star, d_star, new_w_star = max(lags, key=lambda x: abs(x[2]))

            truth_map[(src, tgt)] = {
                "true_src": src,
                "true_tgt": tgt,
                "true_lag_star": int(lag_star),
                "true_base_w": float(base_w_star),
                "true_delta_w": float(d_star),
                "true_new_w": float(new_w_star),
                "true_abs_delta_w": float(abs(d_star)),
                "true_flip": int(np.sign(base_w_star) != np.sign(new_w_star)),
                "true_num_lags_changed": int(len(lags)),
                "true_lags_detail": ";".join(
                    [f"lag{lg}:base{bw:+.4f},d{dw:+.4f},new{nw:+.4f}" for (lg,bw,dw,nw) in lags]
                )
            }

            truth_rows.append({
                "src": src,
                "tgt": tgt,
                "lag_star": int(lag_star),
                "base_w_star": float(base_w_star),
                "delta_w_star": float(d_star),
                "new_w_star": float(new_w_star),
                "abs_delta_w_star": float(abs(d_star)),
                "flip_star": int(np.sign(base_w_star) != np.sign(new_w_star)),
                "num_lags_changed": int(len(lags)),
                "lags_detail": truth_map[(src, tgt)]["true_lags_detail"],
            })

    return truth_rows, truth_map


# -------------------------
# NEW FEATURE #1: markdown/latex tables
# -------------------------
def write_markdown_table(rows, out_path, title=None):
    safe_mkdir(os.path.dirname(out_path))
    lines = []
    if title:
        lines.append(f"### {title}\n")
    header = ["rank","src->tgt","score","v0(lag)","v1(lag)","true?","truth(lag*, base, d, new)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"]*len(header)) + " |")

    for r in rows:
        edge = f"{r['src']}→{r['tgt']}"
        v0 = f"{r['v0_signed']:+.4f}(L{r['lag0']})"
        v1 = f"{r['v1_signed']:+.4f}(L{r['lag1']})"
        truth = ""
        if r.get("is_true_change", 0) == 1:
            truth = f"L{r.get('true_lag_star','')}, {r.get('true_base_w_star',''):+.3f}, {r.get('true_delta_w_star',''):+.3f}, {r.get('true_new_w_star',''):+.3f}"
        lines.append("| " + " | ".join([
            str(r["rank"]),
            edge,
            f"{r['score']:.4f}",
            v0,
            v1,
            str(r.get("is_true_change", "")),
            truth
        ]) + " |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def write_latex_table(rows, out_path, caption=None, label=None):
    safe_mkdir(os.path.dirname(out_path))
    cap = caption or "Top-K change edges"
    lab = label or "tab:topk_change"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{r c r c c c c}")
    lines.append(r"\hline")
    lines.append(r"Rank & Edge ($i\to j$) & Score & $v_0$ & $v_1$ & True & Truth$(\ell^*,w,\Delta,w')$ \\")
    lines.append(r"\hline")
    for r in rows:
        edge = f"{r['src']}\\to{r['tgt']}"
        v0 = f"{r['v0_signed']:+.3f}(L{r['lag0']})"
        v1 = f"{r['v1_signed']:+.3f}(L{r['lag1']})"
        trueflag = str(int(r.get("is_true_change", 0)))
        truth = ""
        if int(r.get("is_true_change", 0)) == 1:
            truth = f"L{r.get('true_lag_star','')}, {r.get('true_base_w_star',0.0):+.2f}, {r.get('true_delta_w_star',0.0):+.2f}, {r.get('true_new_w_star',0.0):+.2f}"
        lines.append(f"{r['rank']} & ${edge}$ & {r['score']:.3f} & {v0} & {v1} & {trueflag} & {truth} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{cap}}}")
    lines.append(rf"\label{{{lab}}}")
    lines.append(r"\end{table}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -------------------------
# NEW FEATURE #2: heatmap figures
# -------------------------
def plot_heatmap_truth_vs_pred(adj_true_tgt_src, adj_pred_tgt_src, out_path, title):
    safe_mkdir(os.path.dirname(out_path))
    # Make 2-panel figure: truth and prediction
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    im1 = ax1.imshow(adj_true_tgt_src, vmin=0, vmax=1)
    ax1.set_title("True change (adj_change)")
    ax1.set_xlabel("src i")
    ax1.set_ylabel("tgt j")

    im2 = ax2.imshow(adj_pred_tgt_src, vmin=0, vmax=1)
    ax2.set_title("Pred top-K change")
    ax2.set_xlabel("src i")
    ax2.set_ylabel("tgt j")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_heatmap_confusion(adj_true_tgt_src, adj_pred_tgt_src, out_path, title):
    """
    confusion map values:
      0: TN
      1: FP
      2: FN
      3: TP
    """
    safe_mkdir(os.path.dirname(out_path))
    N = adj_true_tgt_src.shape[0]
    # exclude diag in the confusion by forcing TN
    true = adj_true_tgt_src.copy().astype(int)
    pred = adj_pred_tgt_src.copy().astype(int)
    for k in range(N):
        true[k, k] = 0
        pred[k, k] = 0

    conf = np.zeros_like(true, dtype=int)
    conf[(true == 0) & (pred == 0)] = 0  # TN
    conf[(true == 0) & (pred == 1)] = 1  # FP
    conf[(true == 1) & (pred == 0)] = 2  # FN
    conf[(true == 1) & (pred == 1)] = 3  # TP

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(conf, vmin=0, vmax=3)
    ax.set_title("Confusion map: 0 TN, 1 FP, 2 FN, 3 TP")
    ax.set_xlabel("src i")
    ax.set_ylabel("tgt j")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------
# main eval + export
# -------------------------
def run(prefix, data_dir="synthetic_step3_v2", out_subdir="exports_step3_v2"):
    adj_change = np.load(os.path.join(data_dir, "adj_change.npy"))  # (tgt,src)
    adj_base   = np.load(os.path.join(data_dir, "adj_base.npy"))    # (tgt,src)
    K = int(adj_change.sum())
    N = adj_change.shape[0]

    truth_rows, truth_map = build_truth_change_table(data_dir)

    v0 = np.load(os.path.join(data_dir, f"{prefix}_regime0_val_matrix.npy"))
    v1 = np.load(os.path.join(data_dir, f"{prefix}_regime1_val_matrix.npy"))
    a0 = np.load(os.path.join(data_dir, f"{prefix}_regime0_adj_hat.npy"))
    a1 = np.load(os.path.join(data_dir, f"{prefix}_regime1_adj_hat.npy"))

    s0, lag0 = best_signed_val_over_lags(v0)  # (src,tgt)
    s1, lag1 = best_signed_val_over_lags(v1)

    cand_true, cand_pred = build_candidate_masks(adj_base, a0, a1)
    score_mag, score_flip = make_scores(s0, s1)

    settings = [
        ("TRUE-mask", cand_true, "magdiff",  score_mag),
        ("TRUE-mask", cand_true, "signflip", score_flip),
        ("PRED-mask", cand_pred, "magdiff",  score_mag),
        ("PRED-mask", cand_pred, "signflip", score_flip),
    ]

    print(f"\n=== {prefix.upper()} change detection (val-diff) + export PLUS++ ===")
    print(f"K(true_change_edges)={K}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = safe_mkdir(os.path.join(data_dir, out_subdir))

    # export truth
    truth_csv = os.path.join(out_dir, f"truth_change_edges_{timestamp}.csv")
    export_csv(truth_rows, truth_csv)
    print(f"[OK] Saved TRUTH CSV: {truth_csv}")

    # also plot truth heatmap once
    truth_fig = os.path.join(out_dir, f"heatmap_truth_adj_change_{timestamp}.png")
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(adj_change, vmin=0, vmax=1)
    ax.set_title("True change edges (adj_change)")
    ax.set_xlabel("src i")
    ax.set_ylabel("tgt j")
    fig.tight_layout()
    fig.savefig(truth_fig, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved truth heatmap: {truth_fig}")

    all_rows = []
    summary_rows = []


    for mask_name, cand_mask, score_name, score in settings:
        edges = topk_edges(score, cand_mask.copy(), K)
        pred = edges_to_pred_adj(edges, N)

        tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_change, pred)
        print(f"[{mask_name:8s} {score_name:8s}] TP={tp} FP={fp} FN={fn} | "
              f"Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

        summary_rows.append({
            "prefix": prefix,
            "mask": mask_name,
            "score_type": score_name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Prec": prec,
            "Rec": rec,
            "F1": f1,
            "SHD": shd,
            "K_true_change": K
        })

        # build topK rows enriched with truth info
        topk_rows_for_tables = []
        for rank, (src, tgt, sc) in enumerate(edges, start=1):
            v0_st = float(s0[src, tgt]); v1_st = float(s1[src, tgt])
            l0 = int(lag0[src, tgt]); l1 = int(lag1[src, tgt])
            is_true = int(adj_change[tgt, src])

            tinfo = truth_map.get((src, tgt), None)
            row = {
                "prefix": prefix,
                "mask": mask_name,
                "score_type": score_name,
                "rank": rank,
                "src": src,
                "tgt": tgt,
                "score": float(sc),
                "v0_signed": v0_st,
                "v1_signed": v1_st,
                "lag0": l0,
                "lag1": l1,
                "is_true_change": is_true,
            }
            if tinfo is not None:
                row.update({
                    "true_lag_star": tinfo["true_lag_star"],
                    "true_base_w_star": tinfo["true_base_w"],
                    "true_delta_w_star": tinfo["true_delta_w"],
                    "true_new_w_star": tinfo["true_new_w"],
                    "true_lags_detail": tinfo["true_lags_detail"],
                })
            topk_rows_for_tables.append(row)
            all_rows.append(row)

        # summary row for CSV
        all_rows.append({
            "prefix": prefix,
            "mask": mask_name,
            "score_type": score_name,
            "rank": 0,
            "TP": tp, "FP": fp, "FN": fn,
            "Prec": prec, "Rec": rec, "F1": f1, "SHD": shd,
            "K_true_change": K
        })

        # ---- NEW: write markdown + latex table files per setting ----
        tag = f"{prefix}_{mask_name}_{score_name}".replace("-", "").replace(" ", "")
        md_path = os.path.join(out_dir, f"topk_table_{tag}_{timestamp}.md")
        tex_path = os.path.join(out_dir, f"topk_table_{tag}_{timestamp}.tex")
        title = f"{prefix.upper()} | {mask_name} | {score_name} | TP={tp}, FP={fp}, FN={fn}, F1={f1:.3f}"
        write_markdown_table(topk_rows_for_tables, md_path, title=title)
        write_latex_table(
            topk_rows_for_tables,
            tex_path,
            caption=title,
            label=f"tab:{tag}_{timestamp}"
        )
        print(f"[OK] Saved tables: {md_path} , {tex_path}")

        # ---- NEW: heatmap truth vs pred + confusion map per setting ----
        fig_out = os.path.join(out_dir, f"heatmap_truth_vs_pred_{tag}_{timestamp}.png")
        plot_heatmap_truth_vs_pred(adj_change, pred, fig_out, title=title)
        conf_out = os.path.join(out_dir, f"heatmap_confusion_{tag}_{timestamp}.png")
        plot_heatmap_confusion(adj_change, pred, conf_out, title=title)
        print(f"[OK] Saved heatmaps: {fig_out} , {conf_out}")

    # export all topK+ rows to CSV
    out_csv = os.path.join(out_dir, f"change_topk_plusplus_{prefix}_{timestamp}.csv")
    export_csv(all_rows, out_csv)
    print(f"[OK] Saved TOPK++ CSV: {out_csv}")
        # ---- NEW: write summary tables (csv/md/tex) ----
    summary_csv = os.path.join(out_dir, f"summary_results_{prefix}_{timestamp}.csv")
    export_csv(summary_rows, summary_csv)

    summary_md  = os.path.join(out_dir, f"summary_results_{prefix}_{timestamp}.md")
    write_summary_markdown(summary_rows, summary_md, title=f"Summary ({prefix.upper()})")

    summary_tex = os.path.join(out_dir, f"summary_results_{prefix}_{timestamp}.tex")
    write_summary_latex(summary_rows, summary_tex, caption=f"Summary ({prefix.upper()})",
                        label=f"tab:summary_{prefix}_{timestamp}")

    print(f"[OK] Saved SUMMARY: {summary_csv} , {summary_md} , {summary_tex}")
    return summary_rows



def main():
    data_dir = "synthetic_step3_v2"
    out_subdir = "exports_step3_v2"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = safe_mkdir(os.path.join(data_dir, out_subdir))

    sum_p = run("parcorr", data_dir=data_dir, out_subdir=out_subdir)
    sum_c = run("cmiknn", data_dir=data_dir, out_subdir=out_subdir)
    merged = sum_p + sum_c

    merged_csv = os.path.join(out_dir, f"summary_results_merged_{timestamp}.csv")
    export_csv(merged, merged_csv)

    merged_md  = os.path.join(out_dir, f"summary_results_merged_{timestamp}.md")
    write_summary_markdown(merged, merged_md, title="Summary (Merged: ParCorr vs CMIknn)")

    merged_tex = os.path.join(out_dir, f"summary_results_merged_{timestamp}.tex")
    write_summary_latex(merged, merged_tex, caption="Summary (Merged: ParCorr vs CMIknn)",
                        label=f"tab:summary_merged_{timestamp}")

    print(f"[OK] Saved MERGED SUMMARY: {merged_csv} , {merged_md} , {merged_tex}")



if __name__ == "__main__":
    main()
