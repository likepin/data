from __future__ import annotations

from pathlib import Path

from posthoc_ecl96_deltaA_manual_gate import find_result_dirs

from .profiles import RESULT_ROOT


def load_result_dirs(pattern: str, pred_file: str, true_file: str) -> list[Path]:
    return find_result_dirs(RESULT_ROOT, pattern, pred_file=pred_file, true_file=true_file)


def try_load_result_dirs(pattern: str, pred_file: str, true_file: str) -> list[Path] | None:
    try:
        return load_result_dirs(pattern=pattern, pred_file=pred_file, true_file=true_file)
    except FileNotFoundError:
        return None
