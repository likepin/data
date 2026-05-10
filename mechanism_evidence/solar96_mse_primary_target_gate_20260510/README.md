# Solar-96 MSE-Primary Target-Gated Dynamic Route

Purpose:
- Freeze `MSE-primary target-gated dynamic route` as a secondary Stage3 route, separate from the strict CACI double-guard route.
- Preserve the main strict route while documenting an MSE-sensitive route for Solar-96 volatility/risk applications.

Selection rule:
- Validation MSE is the primary selector.
- MAE is retained as an audit/non-degradation readout rather than a hard double guard.
- Test is evaluated once for the validation-selected route.
- Shuffle controls break gamma time alignment or target alignment to test whether the gain is route-specific.

Key results:
- Strict target gate selected `stage2_anchor` and therefore reports the adaptive anchor: `0.196010 / 0.227003`.
- MSE-primary target gate selected `target_gate_g10_d20` (`gamma_active_ratio=0.10`, `dynamic_active_ratio=0.20`), test `0.195569 / 0.226831`, gain vs adaptive `+0.2252% / +0.0756%`.

Controls:
- `shuffle_gamma` median: `0.195695 / 0.226860`, gain vs adaptive `+0.1604% / +0.0631%`.
- `shuffle_target` median: `0.195814 / 0.226917`, gain vs adaptive `+0.1001% / +0.0380%`.

Interpretation:
- `MSE-primary` is not a replacement for strict CACI; it is a secondary route for MSE-sensitive settings.
- The observed route beats both shuffle controls, but controls retain some positive gain, so the evidence supports weak target/gamma specificity plus sparse dynamic regularization rather than a pure causal-localization claim.

Files:
- `solar96_mse_primary_target_gate_frozen_table.csv/md`: frozen route table.
- `solar96_mse_primary_target_gate_controls.csv`: shuffle control summary.
- `manifest.json`: source outputs and raw run references.
