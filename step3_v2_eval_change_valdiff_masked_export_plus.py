import os
import numpy as np
import csv
from datetime import datetime

# -------------------------
# helpers
# -------------------------
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
    # remove diagonal
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
    # union header
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    header = sorted(keys)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

# -------------------------
# truth extraction
# -------------------------
def build_truth_change_table(data_dir):
    """
    Returns:
      truth_rows: list dicts for CSV
      truth_map: dict key=(src,tgt) -> dict with DeltaA info aggregated over lags
    """
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

            # collect per-lag info
            lags = []
            for l in range(L):
                d = float(DeltaA[l, tgt, src])
                if abs(d) > 1e-8:
                    base_w = float(A_base[l, tgt, src])
                    new_w  = float(A_base[l, tgt, src] + DeltaA[l, tgt, src])
                    lags.append((l + 1, base_w, d, new_w))

            # aggregate
            # choose lag with max |Delta|
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
                "true_lags_detail": ";".join([f"lag{lg}:base{bw:+.4f},d{dw:+.4f},new{nw:+.4f}" for (lg,bw,dw,nw) in lags])
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
# main eval + export
# -------------------------
def run(prefix, data_dir="synthetic_step3_v2", out_subdir="exports_step3_v2"):
    adj_change = np.load(os.path.join(data_dir, "adj_change.npy"))  # (tgt,src)
    adj_base   = np.load(os.path.join(data_dir, "adj_base.npy"))    # (tgt,src)
    K = int(adj_change.sum())
    N = adj_change.shape[0]

    # truth export (once per run is fine)
    truth_rows, truth_map = build_truth_change_table(data_dir)

    # model outputs
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

    print(f"\n=== {prefix.upper()} change detection (val-diff) + export PLUS ===")
    print(f"K(true_change_edges)={K}")

    all_rows = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(data_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # export truth once (per script execution)
    truth_csv = os.path.join(out_dir, f"truth_change_edges_{timestamp}.csv")
    export_csv(truth_rows, truth_csv)
    print(f"[OK] Saved TRUTH CSV: {truth_csv}")

    for mask_name, cand_mask, score_name, score in settings:
        edges = topk_edges(score, cand_mask.copy(), K)
        pred = edges_to_pred_adj(edges, N)

        tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_change, pred)
        print(f"[{mask_name:8s} {score_name:8s}] TP={tp} FP={fp} FN={fn} | "
              f"Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

        print("  top-K edges (src->tgt): score, v0(lag), v1(lag), is_true_change, (truth lag*, base, delta, new)")
        for rank, (src, tgt, sc) in enumerate(edges, start=1):
            v0_st = float(s0[src, tgt]); v1_st = float(s1[src, tgt])
            l0 = int(lag0[src, tgt]); l1 = int(lag1[src, tgt])
            is_true = int(adj_change[tgt, src])

            tinfo = truth_map.get((src, tgt), None)
            if tinfo is None:
                truth_lag = ""
                truth_base = ""
                truth_d = ""
                truth_new = ""
                truth_flip = ""
                truth_absd = ""
                truth_detail = ""
            else:
                truth_lag = tinfo["true_lag_star"]
                truth_base = tinfo["true_base_w"]
                truth_d = tinfo["true_delta_w"]
                truth_new = tinfo["true_new_w"]
                truth_flip = tinfo["true_flip"]
                truth_absd = tinfo["true_abs_delta_w"]
                truth_detail = tinfo["true_lags_detail"]

            print(f"   #{rank:02d} {src:2d}->{tgt:2d}  score={sc:8.4f}  "
                  f"v0={v0_st:+.5f}(lag{l0})  v1={v1_st:+.5f}(lag{l1})  true={is_true}  "
                  f"| lag*={truth_lag} base={truth_base} d={truth_d} new={truth_new} flip={truth_flip}")

            row = {
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
                "signflip_pred": int(np.sign(v0_st) != np.sign(v1_st)),

                "is_true_change": is_true,

                # truth info if hit
                "true_lag_star": truth_lag,
                "true_base_w_star": truth_base,
                "true_delta_w_star": truth_d,
                "true_new_w_star": truth_new,
                "true_abs_delta_w_star": truth_absd,
                "true_flip_star": truth_flip,
                "true_lags_detail": truth_detail,
            }
            all_rows.append(row)

        # summary row
        all_rows.append({
            "prefix": prefix,
            "mask": mask_name,
            "score_type": score_name,
            "rank": 0,
            "TP": tp, "FP": fp, "FN": fn,
            "Prec": prec, "Rec": rec, "F1": f1, "SHD": shd,
            "K_true_change": K
        })

    out_csv = os.path.join(out_dir, f"change_topk_plus_{prefix}_{timestamp}.csv")
    export_csv(all_rows, out_csv)
    print(f"[OK] Saved TOPK+ CSV: {out_csv}")

def main():
    run("parcorr")
    run("cmiknn")

if __name__ == "__main__":
    main()
