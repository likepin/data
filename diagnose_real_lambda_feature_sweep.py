from __future__ import annotations

import argparse
from pathlib import Path

import diagnose_etth1_lambda_feature_sweep as sweep
from real_dataset_io import count_data_rows, normalize_date_col, normalize_header_mode


REPO = Path(r"C:\Users\cyl\Desktop\iTransformer-phasec-clean")
DATASET_DIR = REPO / "dataset"
RESULT_ROOT = REPO / "results"
AUDIT_ROOT = Path(r"C:\Users\cyl\Desktop\data\deltaA_signal_audit")


PROFILES = {
    "etth196_baseline": {
        "data_csv": DATASET_DIR / "ETTh1.csv",
        "result_pattern": "etth196_validate_baseline_itr3_*projection_*",
        "output_prefix": "etth196",
        "out_dir": AUDIT_ROOT / "etth196_lambda_feature_sweep",
        "split": "ett_hour",
        "date_col": "date",
        "header_mode": "infer",
        "sep": ",",
    },
    "weather96_static": {
        "data_csv": DATASET_DIR / "weather.csv",
        "result_pattern": "weather_96_96_staticcausal_softmax_itr3_*projection_*",
        "output_prefix": "weather96_static",
        "out_dir": AUDIT_ROOT / "weather96_lambda_feature_sweep",
        "split": "custom_ratio",
        "date_col": "date",
        "header_mode": "infer",
        "sep": ",",
    },
    "ecl96_baseline": {
        "data_csv": DATASET_DIR / "ECL.csv",
        "result_pattern": "ecl96_confirm_lr5e4_baseline_itr3_*projection_*",
        "output_prefix": "ecl96_baseline",
        "out_dir": AUDIT_ROOT / "ecl96_lambda_feature_sweep",
        "split": "custom_ratio",
        "date_col": "date",
        "header_mode": "infer",
        "sep": ",",
    },
    "ecl96_static": {
        "data_csv": DATASET_DIR / "ECL.csv",
        "result_pattern": "ecl96_confirm_lr5e4_static_anchor_itr3_*projection_*",
        "output_prefix": "ecl96_static",
        "out_dir": AUDIT_ROOT / "ecl96_lambda_feature_sweep_static",
        "split": "custom_ratio",
        "date_col": "date",
        "header_mode": "infer",
        "sep": ",",
    },
    "traffic96_static": {
        "data_csv": DATASET_DIR / "traffic" / "traffic.csv",
        "result_pattern": "traffic_96_96_staticcausal_softmax_itr3_*projection_*",
        "output_prefix": "traffic96_static",
        "out_dir": AUDIT_ROOT / "traffic96_lambda_feature_sweep",
        "split": "custom_ratio",
        "date_col": "date",
        "header_mode": "infer",
        "sep": ",",
    },
    "solar96_static": {
        "data_csv": DATASET_DIR / "Solar" / "solar_AL.txt",
        "result_pattern": "solar_96_96_staticcausal_softmax_itr3_*projection_*",
        "output_prefix": "solar96_static",
        "out_dir": AUDIT_ROOT / "solar96_lambda_feature_sweep",
        "split": "custom_ratio",
        "date_col": None,
        "header_mode": "none",
        "sep": ",",
    },
}


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def count_csv_rows(path: Path, header_mode: str | None = "infer") -> int:
    return count_data_rows(path, header_mode=header_mode)


def configure_split(split: str, total_rows: int, train_ratio: float, seq_len: int) -> None:
    if split == "ett_hour":
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = train_end + 8 * 30 * 24
    elif split == "custom_ratio":
        train_end = int(total_rows * train_ratio)
        test_len = int(total_rows * 0.2)
        val_end = total_rows - test_len
        test_end = total_rows
    else:
        raise ValueError(f"Unsupported split: {split}")

    if not (seq_len < train_end < val_end < test_end <= total_rows):
        raise ValueError(
            f"Invalid split boundaries: split={split}, total={total_rows}, "
            f"train_end={train_end}, val_end={val_end}, test_end={test_end}, seq_len={seq_len}"
        )
    sweep.TRAIN_END = train_end
    sweep.VAL_END = val_end
    sweep.TEST_END = test_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lambda feature sweep on a configured real-data profile.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--data-csv", type=Path, default=None)
    parser.add_argument("--result-pattern", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--header-mode", choices=["infer", "none"], default=None)
    parser.add_argument("--sep", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--windows", default="20,40,60,80,120")
    parser.add_argument("--ks", default="2,3,5,8")
    parser.add_argument(
        "--modes",
        default="current,change_half,change_slope,change_slope_no_range,level_shift,volatility,tail_risk",
    )
    parser.add_argument("--top-test-configs", type=int, default=20)
    parser.add_argument("--validation-folds", type=int, default=4)
    parser.add_argument(
        "--lambda-scale",
        choices=["legacy_clipped", "unclipped_linear", "log_tail_adaptive"],
        default="legacy_clipped",
    )
    parser.add_argument("--tail-target-width", type=float, default=0.10)
    parser.add_argument("--tail-alpha-min", type=float, default=0.02)
    parser.add_argument("--tail-alpha-max", type=float, default=0.20)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    data_csv = args.data_csv or profile["data_csv"]
    out_dir = args.out_dir or profile["out_dir"]
    output_prefix = args.output_prefix or profile["output_prefix"]
    result_pattern = args.result_pattern or profile["result_pattern"]
    date_col = normalize_date_col(args.date_col if args.date_col is not None else profile.get("date_col", "date"))
    header_mode = normalize_header_mode(args.header_mode or profile.get("header_mode", "infer"))
    sep = args.sep or str(profile.get("sep", ","))
    total_rows = count_csv_rows(data_csv, header_mode=header_mode)

    sweep.DATA_CSV = data_csv
    sweep.RESULT_ROOT = RESULT_ROOT
    sweep.OUT_DIR = out_dir
    sweep.BASELINE_PATTERN = result_pattern
    sweep.OUTPUT_PREFIX = output_prefix
    sweep.DATA_DATE_COL = date_col
    sweep.DATA_HEADER_MODE = header_mode
    sweep.DATA_SEP = sep
    sweep.SEQ_LEN = int(args.seq_len)
    sweep.PRED_LEN = int(args.pred_len)
    sweep.WINDOWS = parse_int_list(args.windows)
    sweep.KS = parse_int_list(args.ks)
    sweep.MODES = parse_str_list(args.modes)
    sweep.TOP_TEST_CONFIGS = int(args.top_test_configs)
    sweep.VALIDATION_FOLDS = int(args.validation_folds)
    sweep.LAMBDA_SCALE = str(args.lambda_scale)
    sweep.TAIL_TARGET_WIDTH = float(args.tail_target_width)
    sweep.TAIL_ALPHA_MIN = float(args.tail_alpha_min)
    sweep.TAIL_ALPHA_MAX = float(args.tail_alpha_max)
    configure_split(
        split=str(profile["split"]),
        total_rows=total_rows,
        train_ratio=float(args.train_ratio),
        seq_len=sweep.SEQ_LEN,
    )

    print(
        "[Profile] "
        f"name={args.profile} rows={total_rows} data={data_csv} "
        f"date_col={date_col} header_mode={header_mode} sep={sep} "
        f"pattern={result_pattern} train_end={sweep.TRAIN_END} "
        f"val_end={sweep.VAL_END} test_end={sweep.TEST_END} "
        f"windows={sweep.WINDOWS} ks={sweep.KS} modes={sweep.MODES} "
        f"lambda_scale={sweep.LAMBDA_SCALE} tail_target_width={sweep.TAIL_TARGET_WIDTH}",
        flush=True,
    )
    sweep.main()


if __name__ == "__main__":
    main()
