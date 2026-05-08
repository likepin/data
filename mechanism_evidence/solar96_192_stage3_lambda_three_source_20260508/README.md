# Solar96/Solar192 Stage3 Lambda Three-Source Evidence

Boundary: this is a post-hoc dynamic add-on on top of the adaptive-alpha anchor. It is not a new training run.

Key results:
- Solar-96: Stage3 selected `stage3_closed_form_top_alpha_5` with MSE/MAE gain vs adaptive anchor +0.0451%/+0.0250%.
- Solar-192: Stage3 selected `stage2_anchor` with MSE/MAE gain vs adaptive anchor +0.0000%/+0.0000%.

Interpretation:
- Adaptive alpha is the Solar performance anchor.
- Stage3 is weak positive for Solar-96 and bypasses to the anchor for Solar-192 under validation selection.

Files:
- `solar_stage3_lambda_three_source_frozen_table.csv/md`: frozen comparison table.
- `manifest.json`: source run directories and raw summary paths.
