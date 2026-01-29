import os
import json
import argparse

import numpy as np


def find_x_npy(data_dir):
    direct = os.path.join(data_dir, "X.npy")
    if os.path.isfile(direct):
        return direct
    for root, _, files in os.walk(data_dir):
        if "X.npy" in files:
            return os.path.join(root, "X.npy")
    raise FileNotFoundError(f"X.npy not found under: {data_dir}")


def find_lambda_files(data_dir, out_dir):
    candidates = [
        os.path.join(out_dir, "lambda_t.npy"),
        os.path.join(data_dir, "lambda_t.npy"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    newest = None
    newest_mtime = -1
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().startswith("lambda_t") and f.lower().endswith(".npy"):
                p = os.path.join(root, f)
                m = os.path.getmtime(p)
                if m > newest_mtime:
                    newest_mtime = m
                    newest = p
    if newest is None:
        raise FileNotFoundError(f"lambda_t.npy not found under: {data_dir}")
    return newest


def find_valid_mask(data_dir, out_dir):
    candidates = [
        os.path.join(out_dir, "lambda_valid_mask.npy"),
        os.path.join(data_dir, "lambda_valid_mask.npy"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def read_t_switch(data_dir, t_switch_arg):
    if t_switch_arg is not None:
        return int(t_switch_arg)
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "t_switch" in meta:
            return int(meta["t_switch"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--t_switch", type=int, default=None)
    parser.add_argument("--out_name", type=str, default="lambda_indexed.npz")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)

    x_path = find_x_npy(data_dir)
    X = np.load(x_path)
    T = X.shape[0]

    lambda_path = find_lambda_files(data_dir, os.path.join(data_dir, "exports_step4"))
    lambda_t = np.load(lambda_path).reshape(-1)

    valid_mask_path = find_valid_mask(data_dir, os.path.join(data_dir, "exports_step4"))
    if valid_mask_path is not None:
        valid_mask = np.load(valid_mask_path).astype(bool)
    else:
        valid_mask = np.isfinite(lambda_t)

    if lambda_t.shape[0] != T:
        raise ValueError(f"lambda_t length {lambda_t.shape[0]} does not match X length {T}")

    if valid_mask.shape[0] != T:
        raise ValueError(f"valid_mask length {valid_mask.shape[0]} does not match X length {T}")

    t_switch = read_t_switch(data_dir, args.t_switch)
    t_switch_out = int(t_switch) if t_switch is not None else -1

    out_path = os.path.join(out_dir, args.out_name)
    np.savez(
        out_path,
        lambda_t=lambda_t.astype(np.float32),
        valid_mask=valid_mask.astype(np.bool_),
        t_switch=np.int32(t_switch_out),
    )

    print("=== Step4: export lambda for dataloader ===")
    print(f"X: {x_path} shape={X.shape}")
    print(f"lambda_t: {lambda_path}")
    print(f"valid_mask: {valid_mask_path if valid_mask_path else '[derived]'}")
    print(f"t_switch: {t_switch_out}")
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
