# Solar96/Solar192 Adaptive-Alpha Evidence

This package freezes the Solar adaptive-alpha branch after the Solar-192 closed-loop run.

Boundary: this is prediction-level baseline/static fusion, not a new train-time graph model and not a formal dynamic closed-loop success claim.

Key results:
- Solar-96: selected per-variable alpha, test MSE/MAE 0.196010/0.227003, gain vs baseline mean +1.621%/+2.160%, gain vs best single +2.678%/+4.301%.
- Solar-192: selected per-variable alpha, test MSE/MAE 0.233232/0.254287, gain vs baseline mean +1.026%/+1.658%, gain vs best single +3.860%/+2.781%.

Files:
- `solar_adaptive_alpha_frozen_table.csv/md`: frozen cross-horizon table.
- `solar_adaptive_alpha_variable_alpha.csv`: all per-target alpha values for Solar-96 and Solar-192.
- `solar_adaptive_alpha_top_alpha_targets.csv`: top-20 static-anchor targets per horizon.
- `manifest.json`: source output paths.
