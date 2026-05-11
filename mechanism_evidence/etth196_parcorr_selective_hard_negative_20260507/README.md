# ETTh1-96 ParCorr Freeze

Purpose:
- Freeze the ETTh1-96 guarded closed-loop result under the current ParCorr ridgebase_sparse interface.
- Keep the result comparable to the existing CMIknn sparse route without mixing old regen assets into the current protocol.
- Preserve ETTh1 as a hard negative overall while documenting that post-hoc dynamic remains Selective vs static anchor.

Main readout:
- Baseline mean: `0.386865 / 0.404862`.
- Static anchor mean: `0.388214 / 0.406042`; `dBase -0.35% / -0.29%`.
- ParCorr guarded post-hoc: `0.388088 / 0.405779`; `dStatic +0.03% / +0.06%`; `dBase -0.32% / -0.23%`.
- Final mode: `Selective` with `mode_reason=selective_activation`.

Backend comparison:
- CMIknn sparse and ParCorr ridgebase_sparse give the same qualitative verdict: `Selective vs static`, but still below baseline overall.
- ParCorr is slightly more conservative than CMIknn on test: `dStatic MSE +0.032% vs +0.070%`, `dStatic MAE +0.065% vs +0.081%`.

Interpretation:
- ETTh1 should remain a falsification-style case rather than a performance headline.
- The guarded protocol is not dead on ETTh1, because dynamic windows can still pass the double guard selectively.
- However, the gain only repairs a weak static anchor and does not surpass the baseline route.

Files:
- `etth196_route_summary.csv`: baseline/static/parcorr route summary.
- `etth196_backend_comparison.csv`: side-by-side CMIknn vs ParCorr selected summaries.
- `manifest.json`: source files and result directories.
