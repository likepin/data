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


def load_n_from_source(data_dir, n_source):
    if n_source == "X.npy":
        x_path = os.path.join(data_dir, "X.npy")
        if not os.path.isfile(x_path):
            raise FileNotFoundError(f"N_source X.npy not found: {x_path}")
        x = np.load(x_path)
        return int(x.shape[1])
    if n_source == "adj_base.npy":
        adj_path = os.path.join(data_dir, "adj_base.npy")
        if not os.path.isfile(adj_path):
            raise FileNotFoundError(f"N_source adj_base.npy not found: {adj_path}")
        adj = np.load(adj_path)
        return int(adj.shape[0])
    raise ValueError(f"Unsupported N_source: {n_source}")


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
    delta_path = os.path.join(data_dir, "DeltaA.npy")
    if os.path.isfile(delta_path):
        return find_true_change_from_deltaA(data_dir, logs)
    candidates = [
        os.path.join(data_dir, "adj_change_true.npy"),
        os.path.join(data_dir, "adj_change.npy"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return np.load(p), p
    hits = search_files(data_dir, lambda n: ("change_true" in n.lower() and n.lower().endswith(".npy")))
    if hits:
        return np.load(hits[0]), hits[0]
    raise FileNotFoundError("true change adj not found (DeltaA.npy/adj_change_true.npy/adj_change.npy).")


def find_true_change_from_deltaA(data_dir, logs, deltaA_name="DeltaA.npy",
                                 cache_name="adj_change_true_from_deltaA.npy"):
    delta_path = os.path.join(data_dir, deltaA_name)
    if not os.path.isfile(delta_path):
        raise FileNotFoundError(f"{deltaA_name} not found: {delta_path}")
    DeltaA = np.load(delta_path)
    log_append(logs, f"DeltaA shape={DeltaA.shape}")
    if DeltaA.ndim != 3:
        raise ValueError(f"DeltaA must be 3D (L,N,N), got {DeltaA.shape}")
    mask = (np.abs(DeltaA) > 0).any(axis=0).astype(np.int32)
    cache_path = os.path.join(data_dir, cache_name)
    try:
        np.save(cache_path, mask.astype(np.int32))
    except Exception:
        pass
    true_edges = edges_from_adj(mask, diag_excluded=True)
    log_append(logs, f"K_true_from_DeltaA={len(true_edges)}")
    base_path = os.path.join(data_dir, "adj_base.npy")
    if os.path.isfile(base_path):
        adj_base = np.load(base_path)
        violations = [(s, t) for (s, t) in true_edges if adj_base[t, s] == 0]
        log_append(logs, f"violations={len(violations)}")
    return mask, delta_path


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


def find_pred_change_adj(data_dir, cfg, logs):
    from graph_io import load_adj, binarize_adj, assert_orientation

    pred_source = cfg.get("pred_change_source")
    pred_prefix = cfg.get("pred_prefix")
    bin_cfg = cfg.get("binarize", {"mode": "binary"})

    if pred_source not in ("valdiff", "signflip", "adjhat_xor", "topk_csv", "valdiff_on_base"):
        raise ValueError(f"pred_change_source must be one of valdiff/signflip/adjhat_xor/topk_csv, got {pred_source}")

    if pred_source in ("valdiff", "signflip"):
        keyword = "valdiff" if pred_source == "valdiff" else "signflip"
        candidates = search_files(
            data_dir,
            lambda n: (n.lower().endswith(".npy") and keyword in n.lower())
        )
        if pred_prefix:
            candidates = [c for c in candidates if pred_prefix.lower() in os.path.basename(c).lower()]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"{pred_source} npy not found or ambiguous. candidates={candidates}"
            )
        adj = load_adj(candidates[0])
        assert_orientation(adj, convention="tgt_src")
        pred = binarize_adj(adj, **bin_cfg)
        return pred, f"{pred_source}:{os.path.basename(candidates[0])}", pred_prefix, None

    if pred_source == "adjhat_xor":
        if not pred_prefix:
            raise ValueError("pred_prefix is required for adjhat_xor.")
        reg0 = os.path.join(data_dir, f"{pred_prefix}_regime0_adj_hat.npy")
        reg1 = os.path.join(data_dir, f"{pred_prefix}_regime1_adj_hat.npy")
        if not (os.path.isfile(reg0) and os.path.isfile(reg1)):
            candidates0 = search_files(data_dir, lambda n: n.lower().endswith("_regime0_adj_hat.npy"))
            candidates1 = search_files(data_dir, lambda n: n.lower().endswith("_regime1_adj_hat.npy"))
            raise FileNotFoundError(
                f"adjhat_xor files not found for prefix={pred_prefix}. "
                f"candidates0={candidates0}, candidates1={candidates1}"
            )
        adj0 = load_adj(reg0)
        adj1 = load_adj(reg1)
        assert_orientation(adj0, convention="tgt_src")
        assert_orientation(adj1, convention="tgt_src")
        adj0_bin = binarize_adj(adj0, **bin_cfg)
        adj1_bin = binarize_adj(adj1, **bin_cfg)
        pred = (adj0_bin != adj1_bin).astype(np.int32)
        return pred, f"adjhat_xor:{pred_prefix}", pred_prefix, None

    if pred_source == "valdiff_on_base":
        adj_base, _ = find_base_adj(data_dir, logs)
        if adj_base is None:
            raise FileNotFoundError("adj_base not found for valdiff_on_base.")
        top_k = int(cfg.get("top_k", len(edges_from_adj(adj_base, diag_excluded=True))))
        prefix = cfg.get("pred_prefix", "cmiknn")
        pred, scores = predict_change_valdiff_on_base(data_dir, prefix, adj_base, top_k, logs)
        return pred, f"valdiff_on_base:{prefix}", prefix, scores

    # topk_csv
    n_source = cfg.get("N_source", "X.npy")
    N = load_n_from_source(data_dir, n_source)
    csvs = search_files(data_dir, lambda n: n.lower().startswith("change_topk_plusplus") and n.lower().endswith(".csv"))
    if pred_prefix:
        csvs = [c for c in csvs if pred_prefix.lower() in os.path.basename(c).lower()]
    if len(csvs) != 1:
        raise FileNotFoundError(f"topk_csv not found or ambiguous. candidates={csvs}")
    pred, scores = load_edges_from_topk_csv(csvs[0], N)
    return pred, f"topk_csv:{os.path.basename(csvs[0])}", pred_prefix, scores


def load_edges_from_topk_csv(csv_path, N):
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
    edges = set()
    scores = {}
    for r in rows:
        try:
            src = int(float(r.get("src")))
            tgt = int(float(r.get("tgt")))
            sc = float(r.get("score", 1.0))
        except Exception:
            continue
        if src == tgt:
            continue
        if src < 0 or tgt < 0 or src >= N or tgt >= N:
            raise ValueError(f"Edge out of range in {csv_path}: src={src}, tgt={tgt}, N={N}")
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


def predict_change_valdiff_on_base(data_dir, prefix, adj_base, top_k, logs, score_name="valdiff"):
    v0_path = os.path.join(data_dir, f"{prefix}_regime0_val_matrix.npy")
    v1_path = os.path.join(data_dir, f"{prefix}_regime1_val_matrix.npy")
    if not (os.path.isfile(v0_path) and os.path.isfile(v1_path)):
        raise FileNotFoundError(f"val_matrix not found for prefix {prefix}")
    val0 = np.load(v0_path)
    val1 = np.load(v1_path)
    score = np.abs(val1 - val0)
    if score.ndim != 3:
        raise ValueError(f"val_matrix must be 3D (src,tgt,lag), got {score.shape}")
    score2d = score.max(axis=2)
    base_mask_src_tgt = (adj_base.T != 0).astype(np.float32)
    score2d = score2d * base_mask_src_tgt
    np.fill_diagonal(score2d, 0.0)
    flat = np.argsort(score2d.reshape(-1))[::-1]
    N = score2d.shape[0]
    edges = []
    for idx in flat:
        src = idx // N
        tgt = idx % N
        if score2d[src, tgt] <= 0:
            continue
        if adj_base[tgt, src] == 0:
            continue
        edges.append((src, tgt))
        if len(edges) >= top_k:
            break
    pred = np.zeros((N, N), dtype=np.int32)
    scores = {}
    for (src, tgt) in edges:
        pred[tgt, src] = 1
        scores[(src, tgt)] = float(score2d[src, tgt])
    log_append(logs, f"K_pred_from_valdiff_on_base={len(edges)}")
    return pred, scores


def gated_change_from_deltaA(adj_base, DeltaA, gate_weight):
    base = np.array(adj_base)
    delta = np.array(DeltaA)
    if delta.ndim != 3:
        raise ValueError("DeltaA must be (L,N,N)")
    if base.ndim != 3:
        raise ValueError("A_base must be (L,N,N)")
    delta_sum = delta.sum(axis=0)
    base_sum = base.sum(axis=0)
    adj_reg1 = base_sum + gate_weight * delta_sum
    base_bin = (np.abs(base_sum) > 1e-8).astype(np.int32)
    reg1_bin = (np.abs(adj_reg1) > 1e-8).astype(np.int32)
    change = (reg1_bin != base_bin).astype(np.int32)
    return change


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
