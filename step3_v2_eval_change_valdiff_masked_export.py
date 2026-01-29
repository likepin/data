import os
import numpy as np
import csv
from datetime import datetime

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
            lag_out[src, tgt] = k + 1  # convert to lag in 1..tau_max
    return out, lag_out  # (src,tgt), signed, and chosen lag

def metrics_excluding_diag(adj_true, adj_hat):
    # adj_true/adj_hat are in [tgt,src] convention
    N = adj_true.shape[0]
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    y_true = adj_true[mask].astype(int)
    y_hat = adj_hat[mask].astype(int)
    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    shd  = int(np.abs(y_true - y_hat).sum())
    return tp, fp, fn, prec, rec, f1, shd

def build_candidate_masks(adj_base_tgt_src, a0_tgt_src, a1_tgt_src):
    # convert to (src,tgt) boolean masks
    cand_true = (adj_base_tgt_src.T == 1)               # known true base edges
    cand_pred = ((a0_tgt_src == 1) & (a1_tgt_src == 1)).T  # predicted common edges
    return cand_true, cand_pred

def make_scores(s0, s1):
    # s0/s1 are signed best-over-lags values, shape (src,tgt)
    score_mag = np.abs(np.abs(s1) - np.abs(s0))
    sign_flip = (np.sign(s0) != np.sign(s1)).astype(np.float32)
    score_flip = sign_flip * (np.abs(s0) + np.abs(s1))  # prioritize strong + flipped
    return score_mag, score_flip

def topk_edges(score, candidate_mask, K):
    """
    score: (src,tgt), candidate_mask: (src,tgt) bool
    return list of (src,tgt,score) length K (or fewer if not enough candidates)
    """
    N = score.shape[0]
    score2 = score.copy()

    # remove diagonal and non-candidates
    for k in range(N):
        candidate_mask[k, k] = False
    score2[~candidate_mask] = -np.inf

    flat = np.argsort(score2.reshape(-1))[::-1]
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
    # edges are (src,tgt,score), produce pred adjacency in [tgt,src]
    pred = np.zeros((N, N), dtype=np.int32)
    for src, tgt, _ in edges:
        pred[tgt, src] = 1
    return pred

def export_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def run(prefix, data_dir="synthetic_step3_v2", out_subdir="exports_step3_v2"):
    # Load truth
    adj_change = np.load(os.path.join(data_dir, "adj_change.npy"))  # [tgt,src]
    adj_base   = np.load(os.path.join(data_dir, "adj_base.npy"))    # [tgt,src]
    K = int(adj_change.sum())
    N = adj_change.shape[0]

    # Load model outputs (need val + adj)
    v0 = np.load(os.path.join(data_dir, f"{prefix}_regime0_val_matrix.npy"))
    v1 = np.load(os.path.join(data_dir, f"{prefix}_regime1_val_matrix.npy"))
    a0 = np.load(os.path.join(data_dir, f"{prefix}_regime0_adj_hat.npy"))
    a1 = np.load(os.path.join(data_dir, f"{prefix}_regime1_adj_hat.npy"))

    s0, lag0 = best_signed_val_over_lags(v0)  # (src,tgt)
    s1, lag1 = best_signed_val_over_lags(v1)

    cand_true, cand_pred = build_candidate_masks(adj_base, a0, a1)
    score_mag, score_flip = make_scores(s0, s1)

    settings = [
        ("TRUE-mask", cand_true, "magdiff", score_mag),
        ("TRUE-mask", cand_true, "signflip", score_flip),
        ("PRED-mask", cand_pred, "magdiff", score_mag),
        ("PRED-mask", cand_pred, "signflip", score_flip),
    ]

    print(f"\n=== {prefix.upper()} change detection (val-diff) + export ===")
    print(f"K(true_change_edges)={K}")

    all_rows = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for mask_name, cand_mask, score_name, score in settings:
        edges = topk_edges(score, cand_mask.copy(), K)
        pred = edges_to_pred_adj(edges, N)

        tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_change, pred)
        print(f"[{mask_name:8s} {score_name:8s}] TP={tp} FP={fp} FN={fn} | "
              f"Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

        # Print top-K edges
        print("  top-K edges (src->tgt): score, v0(lag), v1(lag), is_true_change")
        for rank, (src, tgt, sc) in enumerate(edges, start=1):
            v0_st = float(s0[src, tgt]); v1_st = float(s1[src, tgt])
            l0 = int(lag0[src, tgt]); l1 = int(lag1[src, tgt])
            is_true = int(adj_change[tgt, src])  # [tgt,src]
            print(f"   #{rank:02d} {src:2d}->{tgt:2d}  "
                  f"score={sc:8.4f}  v0={v0_st:+.5f}(lag{l0})  v1={v1_st:+.5f}(lag{l1})  true={is_true}")

            all_rows.append({
                "prefix": prefix,
                "mask": mask_name,
                "score_type": score_name,
                "rank": rank,
                "src": src,
                "tgt": tgt,
                "score": sc,
                "v0_signed": v0_st,
                "v1_signed": v1_st,
                "lag0": l0,
                "lag1": l1,
                "abs_v0": abs(v0_st),
                "abs_v1": abs(v1_st),
                "absdiff": abs(abs(v1_st) - abs(v0_st)),
                "signflip": int(np.sign(v0_st) != np.sign(v1_st)),
                "is_true_change": is_true,
            })

        # also save summary row for this setting
        all_rows.append({
            "prefix": prefix,
            "mask": mask_name,
            "score_type": score_name,
            "rank": 0,
            "src": -1,
            "tgt": -1,
            "score": np.nan,
            "v0_signed": np.nan,
            "v1_signed": np.nan,
            "lag0": -1,
            "lag1": -1,
            "abs_v0": np.nan,
            "abs_v1": np.nan,
            "absdiff": np.nan,
            "signflip": -1,
            "is_true_change": -1,
            "TP": tp, "FP": fp, "FN": fn,
            "Prec": prec, "Rec": rec, "F1": f1, "SHD": shd,
        })

    # Write CSV
    out_dir = os.path.join(data_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"change_topk_{prefix}_{timestamp}.csv")
    # ensure consistent header
    # collect union keys
    keys = set()
    for r in all_rows:
        keys |= set(r.keys())
    keys = sorted(keys)
    # normalize rows
    norm_rows = []
    for r in all_rows:
        rr = {k: r.get(k, "") for k in keys}
        norm_rows.append(rr)

    export_csv(norm_rows, out_csv)
    print(f"[OK] Saved CSV: {out_csv}")

def main():
    run("parcorr")
    run("cmiknn")

if __name__ == "__main__":
    main()
