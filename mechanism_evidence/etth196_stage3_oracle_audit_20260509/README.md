# ETTh1-96 Stage3 Dynamic Oracle Audit

Purpose:
- Diagnose whether the current ETTh1 dynamic graph branch contains useful upper-bound signal once the gate is made perfect.
- Keep this separate from method performance: all oracle rows use labels for routing decisions.

Main test readout using `static_p0_dynamic`:
- Window-level oracle: `0.378003 / 0.397727`, gain vs adaptive anchor `+0.5939% / +0.4438%`, active ratio `0.4101`.
- Target-level oracle: `0.377228 / 0.396809`, gain vs adaptive anchor `+0.7976% / +0.6734%`, active ratio `0.3108`.
- Point-level oracle: `0.372907 / 0.390653`, gain vs adaptive anchor `+1.9339% / +2.2144%`, active ratio `0.3296`.

Interpretation guide:
- `oracle_window_gate` is the most realistic upper bound for a lambda/window gate.
- `oracle_target_gate` is the upper bound for target-specific `lambda_i` routing.
- `oracle_point_gate` is an aggressive diagnostic ceiling and should not be treated as an implementable route.

Files:
- `etth196_stage3_oracle_audit_table.csv/md`: full oracle table for `val` and `test`, both dynamic-source variants.
- `manifest.json`: source paths and fixed settings.
