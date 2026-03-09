## PhaseA Rulebook

This document records the current provisional evaluation rule used for the synthetic PhaseA benchmark.

### Rule Status
- Scope: synthetic PhaseA only
- Legacy fields retained: `True`
- Purpose: recover valid main strategies without letting negative controls pass

### Peak Delay v3_v2
- peak_delay_min_abs_thr_v2: `121.750000`
- peak_delay_min_abs_rule_v2: `min(0.65*switch_window,max(default_abs_thr,mapped_shift_q75))`
- peak_delay_min_rel_thr_v2: `121.750000`
- peak_delay_min_rel_rule_v2: `value <= mapped_shift_q75`
- Interpretation: peak delay is treated as a temporal misalignment diagnostic and is calibrated against `shift` controls.

### Current Outcome
- main_runs_pass_rate_v3_v2: `0.667`
- negative_control_pass_rate_v3_v2: `0.000`
- Recommendation: keep this rule as the provisional synthetic benchmark standard and re-validate before transferring to real data.
