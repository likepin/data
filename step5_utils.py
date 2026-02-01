import os
import json
import csv
from datetime import datetime

import numpy as np


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


def log_append(logs, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    logs.append(line)


def write_logs(logs, out_path):
    safe_mkdir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(logs) + "\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def search_files(data_dir, predicate):
    hits = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if predicate(name):
                hits.append(os.path.join(root, name))
    return hits


def load_lambda_and_mask(data_dir, logs):
    best = os.path.join(data_dir, "exports_step4", "best_lambda_t.npy")
    lam = os.path.join(data_dir, "exports_step4", "lambda_t.npy")
    idx = os.path.join(data_dir, "lambda_indexed.npz")

    lambda_path = find_first_existing([best, lam])
    valid_mask = None
    source = None
    t_switch = None

    if lambda_path:
        lambda_t = np.load(lambda_path).reshape(-1)
        source = lambda_path
        vm_path = os.path.join(data_dir, "exports_step4", "lambda_valid_mask.npy")
        if os.path.isfile(vm_path):
            valid_mask = np.load(vm_path).astype(bool)
        else:
            log_append(logs, "WARN: lambda_valid_mask.npy not found, using all-True mask.")
            valid_mask = np.ones_like(lambda_t, dtype=bool)
    elif os.path.isfile(idx):
        npz = np.load(idx)
        lambda_t = np.array(npz["lambda_t"]).reshape(-1)
        source = idx
        if "valid_mask" in npz:
            valid_mask = np.array(npz["valid_mask"]).astype(bool)
        else:
            log_append(logs, "WARN: valid_mask not found in lambda_indexed.npz, using all-True mask.")
            valid_mask = np.ones_like(lambda_t, dtype=bool)
        if "t_switch" in npz:
            t_switch = int(npz["t_switch"])
    else:
        raise FileNotFoundError("lambda_t not found (best_lambda_t.npy/lambda_t.npy/lambda_indexed.npz).")

    # t_switch from meta if not found
    if t_switch is None:
        meta_path = os.path.join(data_dir, "meta.json")
        if os.path.isfile(meta_path):
            meta = read_json(meta_path)
            if "t_switch" in meta:
                t_switch = int(meta["t_switch"])

    return lambda_t, valid_mask, source, t_switch


def detect_lambda_config(data_dir, fallback="(unknown)"):
    cfg = os.path.join(data_dir, "exports_step4", "best_config.json")
    if os.path.isfile(cfg):
        try:
            js = read_json(cfg)
            w = js.get("window", None)
            k = js.get("k", None)
            if w is not None and k is not None:
                return f"({int(w)},{int(k)})"
        except Exception:
            pass
    return fallback


def find_true_change_adj(data_dir, logs):
    candidates = [
        os.path.join(data_dir, "adj_change_true.npy"),
        os.path.join(data_dir, "adj_change.npy"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return np.load(p), p
    # fallback: any file with change_true in name
    hits = search_files(data_dir, lambda n: ("change_true" in n.lower() and n.lower().endswith(".npy")))
    if hits:
        return np.load(hits[0]), hits[0]
    raise FileNotFoundError("true change adj not found (adj_change_true.npy or adj_change.npy).")


def find_base_adj(data_dir, logs):
    candidates = [
        os.path.join(data_dir, "adj_base_true.npy"),
        os.path.join(data_dir, "adj_regime0_true.npy"),
        os.path.join(data_dir, "adj_base.npy"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return np.load(p), p
    return None, None


def edges_from_adj(adj, diag_excluded=True):
    adj = np.array(adj)
    N = adj.shape[0]
    edges = set()
    for tgt in range(N):
        for src in range(N):
            if diag_excluded and src == tgt:
                continue
            if adj[tgt, src] != 0:
                edges.add((src, tgt))
    return edges


def confusion(pred_edges, true_edges):
    tp = len(pred_edges & true_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return tp, fp, fn, prec, rec, f1


def shd_from_edges(pred_edges, true_edges):
    tp, fp, fn, _, _, _ = confusion(pred_edges, true_edges)
    return fp + fn


def find_pred_change_adj(data_dir, logs):
    # 1) direct pred change files
    candidates = search_files(
        data_dir,
        lambda n: (n.lower().endswith(".npy") and ("chg_pred" in n.lower() or "valdiff" in n.lower()))
    )
    for c in candidates:
        if "chg_pred_by_valdiff" in os.path.basename(c).lower():
            return np.load(c), f"valdiff:{os.path.basename(c)}", None, None
    for c in candidates:
        if "chg_pred_by_signflip" in os.path.basename(c).lower():
            return np.load(c), f"signflip:{os.path.basename(c)}", None, None

    # 2) adj_hat XOR
    reg0 = search_files(data_dir, lambda n: n.lower().endswith("_regime0_adj_hat.npy"))
    reg1 = search_files(data_dir, lambda n: n.lower().endswith("_regime1_adj_hat.npy"))
    reg0_map = {os.path.basename(p).replace("_regime0_adj_hat.npy", ""): p for p in reg0}
    reg1_map = {os.path.basename(p).replace("_regime1_adj_hat.npy", ""): p for p in reg1}
    prefixes = sorted(set(reg0_map.keys()) & set(reg1_map.keys()))
    if prefixes:
        prefix = None
        for p in prefixes:
            if "parcorr" in p.lower():
                prefix = p
                break
        if prefix is None:
            prefix = prefixes[0]
        adj0 = np.load(reg0_map[prefix])
        adj1 = np.load(reg1_map[prefix])
        pred = (adj0 != adj1).astype(np.int32)
        return pred, f"adjhat_xor:{prefix}", prefix, None

    # 3) change_topk_plusplus CSV fallback
    csvs = search_files(data_dir, lambda n: n.lower().startswith("change_topk_plusplus") and n.lower().endswith(".csv"))
    if csvs:
        csv_path = sorted(csvs, key=os.path.getmtime)[-1]
        pred, scores = load_edges_from_topk_csv(csv_path)
        return pred, f"topk_csv:{os.path.basename(csv_path)}", None, scores

    raise FileNotFoundError("pred change adj not found (chg_pred_by_* or adj_hat XOR).")


def load_edges_from_topk_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("rank", "") == "":
                continue
            try:
                rank = int(float(r.get("rank")))
            except Exception:
                rank = 0
            if rank <= 0:
                continue
            rows.append(r)
    # prefer PRED-mask + magdiff
    def match_row(r):
        m = str(r.get("mask", "")).lower()
        s = str(r.get("score_type", "")).lower()
        return ("pred" in m) and ("mag" in s)
    picked = [r for r in rows if match_row(r)]
    if not picked:
        picked = rows
    edges = set()
    scores = {}
    for r in picked:
        try:
            src = int(float(r.get("src")))
            tgt = int(float(r.get("tgt")))
            sc = float(r.get("score", 1.0))
        except Exception:
            continue
        edges.add((src, tgt))
        scores[(src, tgt)] = sc
    # build adj
    if not edges:
        return None, None
    N = max(max(i, j) for (i, j) in edges) + 1
    adj = np.zeros((N, N), dtype=np.int32)
    for (src, tgt) in edges:
        adj[tgt, src] = 1
    return adj, scores


def best_signed_val_over_lags(val_matrix):
    vals = val_matrix[:, :, 1:]
    idx = np.argmax(np.abs(vals), axis=2)
    out = np.zeros(vals.shape[:2], dtype=np.float32)
    for src in range(out.shape[0]):
        for tgt in range(out.shape[1]):
            k = int(idx[src, tgt])
            out[src, tgt] = vals[src, tgt, k]
    return out


def load_valdiff_scores(data_dir, prefix, logs):
    if prefix is None:
        return None
    v0_path = os.path.join(data_dir, f"{prefix}_regime0_val_matrix.npy")
    v1_path = os.path.join(data_dir, f"{prefix}_regime1_val_matrix.npy")
    if not (os.path.isfile(v0_path) and os.path.isfile(v1_path)):
        log_append(logs, f"WARN: val_matrix not found for prefix {prefix}, no scores.")
        return None
    v0 = np.load(v0_path)
    v1 = np.load(v1_path)
    s0 = best_signed_val_over_lags(v0)
    s1 = best_signed_val_over_lags(v1)
    score = np.abs(np.abs(s1) - np.abs(s0))
    scores = {}
    N = score.shape[0]
    for src in range(N):
        for tgt in range(N):
            if src == tgt:
                continue
            scores[(src, tgt)] = float(score[src, tgt])
    return scores


def compute_expected_metrics(tp0, fp0, k_true, p_active):
    p = float(max(0.0, min(1.0, p_active)))
    tp = p * tp0
    fp = p * fp0
    fn = float(k_true) - tp
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    shd = fp + fn
    return tp, fp, fn, prec, rec, f1, shd


def quantile_mask(values, valid_mask, q, mode):
    v = values[valid_mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(values, dtype=bool)
    thr = float(np.quantile(v, q))
    if mode == "ge":
        return valid_mask & (values >= thr)
    return valid_mask & (values <= thr)


def mean_over_mask(values, mask):
    vals = values[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def active_fraction_hard(lambda_t, subset_mask, tau):
    if subset_mask.sum() == 0:
        return 0.0, 0
    active = (lambda_t < tau) & subset_mask
    frac = float(active.sum() / subset_mask.sum())
    return frac, int(active.sum())


def active_fraction_soft(lambda_t, subset_mask, w_thresh, mode="mean_w", p_thresh=0.5):
    if subset_mask.sum() == 0:
        return 0.0
    w_t = 1.0 - lambda_t
    w_sub = w_t[subset_mask]
    w_sub = w_sub[np.isfinite(w_sub)]
    if w_sub.size == 0:
        return 0.0
    if mode == "frac_active":
        frac = float((w_sub > w_thresh).sum() / w_sub.size)
        if frac < p_thresh:
            return 0.0
        return frac
    # mean_w
    mean_w = float(np.mean(w_sub))
    if mean_w < w_thresh:
        return 0.0
    return mean_w
