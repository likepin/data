import os
import argparse


KEYWORDS = [
    "DeltaA", "adj_change", "adj_base", "regime0", "regime1",
    "val_matrix", "p_matrix", "chg_pred", "valdiff", "signflip",
    "cmiknn", "parcorr",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_path = args.out_path or os.path.join(data_dir, "exports_step5", "debug_tree.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    hits = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            for k in KEYWORDS:
                if k.lower() in name.lower():
                    hits.append(os.path.join(root, name))
                    break

    hits = sorted(hits)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(hits) + "\n")

    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
