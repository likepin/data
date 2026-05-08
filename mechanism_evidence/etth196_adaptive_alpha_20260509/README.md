# ETTh1-96 Adaptive-Alpha Evidence

Purpose:
- Freeze the lightweight ETTh1-96 adaptive-alpha pilot over the existing baseline/static anchor prediction family.
- Keep this branch separate from the guarded post-hoc closed-loop result, which remains Selective vs static but below the baseline route.
- Reclassify ETTh1 at the route-ablation level: train-time static and guarded post-hoc stay weak, but prediction-level adaptive fusion is now positive overall.

Main readout:
- Baseline prediction-mean ensemble: `0.381221 / 0.400097`.
- Static prediction-mean ensemble: `0.382232 / 0.401138`; `dBase -0.27% / -0.26%`.
- Guarded post-hoc: `0.388088 / 0.405779`; `dStatic +0.03% / +0.06%`; `dBase -1.80% / -1.42%`.
- Adaptive fusion (per-variable shrinkage alpha): `0.380261 / 0.399500`; `dBase +0.25% / +0.15%`; `dStatic +0.52% / +0.41%`; `dBestSingle +2.27% / +1.76%`.
- Route-level projection-mean baselines are frozen separately in the cross-dataset route-ablation table; do not mix those reference numbers with the prediction-mean ensemble rows in this package.

Interpretation:
- ETTh1 is no longer a hard negative overall once prediction-level adaptive fusion is allowed.
- The older guarded post-hoc conclusion still stands locally: dynamic windows can be Selective vs static, but that branch remains weaker than adaptive fusion.
- This result should be treated as a route-ablation update, not as evidence that train-time graph injection itself is strong on ETTh1.

Files:
- `etth196_adaptive_alpha_frozen_table.csv/md`: frozen comparison table.
- `etth196_adaptive_alpha_variable_alpha.csv`: per-target adaptive alpha values.
- `raw_outputs/`: copied small raw summaries from the pilot run.
- `manifest.json`: source paths and selection summary.
