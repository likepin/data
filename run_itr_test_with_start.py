from __future__ import annotations

import runpy
import sys
from pathlib import Path
import os


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_itr_test_with_start.py <path-to-run.py> [run.py args...]")
    run_py = Path(sys.argv[1]).resolve()
    if not run_py.exists():
        raise FileNotFoundError(run_py)
    code = run_py.read_text(encoding="utf-8")
    old = "    else:\n        ii = 0\n        setting = '{}_{}_{}_{}_ft{}"
    new = "    else:\n        ii = args.itr_start\n        setting = '{}_{}_{}_{}_ft{}"
    if old not in code:
        raise RuntimeError("Could not find the test-only ii assignment in run.py")
    patched = code.replace(old, new, 1)
    os.chdir(run_py.parent)
    sys.path.insert(0, str(run_py.parent))
    sys.argv = [str(run_py), *sys.argv[2:]]
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(run_py),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(patched, str(run_py), "exec"), globals_dict)


if __name__ == "__main__":
    main()
