# Traffic96 Stage3 Lambda Three-Source Evidence

This package freezes the Stage3-Pilot result for Traffic-96.

Scope:
- `Stage2 anchor`: adaptive-alpha baseline/static ensemble.
- `Stage3`: add a lambda-gated posthoc dynamic increment on top of the Stage2 anchor.
- Default dynamic source: `static_p0`, matching the existing posthoc closed-loop convention.
- Audit dynamic source: `static_mean`, confirming the result is not projection-0-specific.

Interpretation:
- The default Stage3 result is weak positive over Stage2.
- The test shuffled-gamma negative control is thin.
- Use this as a small dynamic-aware increment, not as a strong dynamic-mainline success.

Frozen table: `performance/stage3_lambda_three_source/tables/traffic96_static_stage3_lambda_three_source_frozen_table.md`
