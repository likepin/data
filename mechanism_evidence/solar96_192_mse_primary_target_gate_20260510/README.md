# Solar-96/192 MSE-Primary Target-Gated Dynamic Route

Purpose:
- Freeze `MSE-primary target-gated dynamic route` as a secondary Stage3 route, separate from the strict CACI double-guard route.
- Compare Solar-96 and Solar-192 under the same strict-vs-MSE-primary protocol.
- Preserve the main strict route while documenting a loss-specific route for MSE-sensitive Solar settings.

Selection rule:
- Strict route keeps MSE/MAE guard plus fold stability and may fall back to the adaptive anchor.
- MSE-primary route selects by validation MSE; MAE is retained as an audit/non-degradation readout.
- Test is evaluated once for the validation-selected route.
- Shuffle controls break gamma time alignment or target alignment to audit route specificity.

Key results:
- Solar-96 strict target gate selected `stage2_anchor`: `0.196010 / 0.227003`, gain vs adaptive `+0.0000% / +0.0000%`.
- Solar-96 MSE-primary best variant `static_p0` selected `target_gate_g10_d20` (`gamma_active_ratio=0.10`, `dynamic_active_ratio=0.20`), test `0.195569 / 0.226831`, gain vs adaptive `+0.2252% / +0.0756%`.
- Solar-192 strict target gate selected `stage2_anchor`: `0.233232 / 0.254287`, gain vs adaptive `+0.0000% / +0.0000%`.
- Solar-192 MSE-primary best variant `static_mean` selected `target_gate_g10_d20` (`gamma_active_ratio=0.10`, `dynamic_active_ratio=0.20`), test `0.233103 / 0.254251`, gain vs adaptive `+0.0551% / +0.0141%`.

Controls:
- Solar-96 `shuffle_gamma` median: `0.195695 / 0.226860`, gain vs adaptive `+0.1604% / +0.0631%`.
- Solar-96 `shuffle_target` median: `0.195814 / 0.226917`, gain vs adaptive `+0.1001% / +0.0380%`.
- Solar-192 `shuffle_gamma` median: `0.233201 / 0.254278`, gain vs adaptive `+0.0131% / +0.0037%`.
- Solar-192 `shuffle_target` median: `0.233184 / 0.254278`, gain vs adaptive `+0.0203% / +0.0037%`.

Interpretation:
- Strict CACI remains the conservative route and falls back on Solar-96/192 under this target-gate design.
- MSE-primary is not a replacement for strict CACI; it is a secondary loss-specific route.
- Solar-96 shows a small but repeatable MSE-sensitive gain; Solar-192 shows a smaller gain that still beats shuffle medians for the best variant.
- Because shuffle controls retain some positive gain, the evidence supports weak target/gamma specificity plus sparse dynamic regularization rather than a pure causal-localization claim.

Files:
- `solar96_192_mse_primary_target_gate_frozen_table.csv/md`: frozen route table.
- `solar96_192_mse_primary_target_gate_controls.csv`: shuffle control summary.
- `manifest.json`: source outputs and raw run references.
