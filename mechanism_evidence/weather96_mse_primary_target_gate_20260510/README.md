# Weather-96 MSE-Primary Target-Gated Dynamic Route

Purpose:
- Freeze Weather-96 as a lightweight boundary check for the MSE-primary target-gated dynamic route.
- Keep strict CACI and MSE-primary route separate.

Key results:
- Adaptive anchor: `0.169801 / 0.210550`. This is stronger than the static-only Weather anchor used in older post-hoc tables.
- Strict target gate selected `stage2_anchor` and fell back to adaptive anchor: `0.169801 / 0.210550`.
- MSE-primary best variant `static_mean` selected `target_gate_g20_d40`, test `0.169689 / 0.210608`, gain vs adaptive `+0.0664% / -0.0276%`.

Controls:
- `shuffle_gamma` median: `0.169809 / 0.210622`, gain vs adaptive `-0.0048% / -0.0341%`.
- `shuffle_target` median: `0.169750 / 0.210595`, gain vs adaptive `+0.0303% / -0.0214%`.

Interpretation:
- Weather confirms the expected boundary behavior: strict route protects the anchor, while MSE-primary admits a small MSE gain at the cost of a small MAE regression.
- Observed MSE gain beats the shuffle medians, but the effect is small and should be reported as loss-specific rather than a headline performance route.

Files:
- `weather96_mse_primary_target_gate_frozen_table.csv/md`: frozen route table.
- `weather96_mse_primary_target_gate_controls.csv`: shuffle control summary.
- `manifest.json`: source outputs and raw run references.
