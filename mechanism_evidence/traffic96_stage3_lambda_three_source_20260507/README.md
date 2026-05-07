# Traffic96 Stage3 Lambda Three-Source Evidence

This package freezes the Stage3-Pilot result for Traffic-96.

Scope:
- `Stage2 anchor`: adaptive-alpha baseline/static ensemble.
- `Stage3`: add a lambda-gated posthoc dynamic increment on top of the Stage2 anchor.
- Default dynamic source: `static_p0`, matching the existing posthoc closed-loop convention.
- Audit dynamic source: `static_mean`, confirming the result is not projection-0-specific.
- Closed-form eta2: validation-estimated eta clipped by `eta_max=2.0`; this is the recommended Stage3 performance anchor for risk-window diagnostics.

Interpretation:
- The default grid Stage3 result is weak positive over Stage2.
- The closed-form eta2 result is slightly better than grid, but still a weak positive increment.
- The test shuffled-gamma negative control is thin.
- Use this as a small dynamic-aware increment, not as a strong dynamic-mainline success.

Frozen table: `performance/stage3_lambda_three_source/tables/traffic96_static_stage3_lambda_three_source_frozen_table.md`
