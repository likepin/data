## Phase A Block-Shuffle Summary

- blockshuffle mean directional_align_overall: `0.459327`
- blockshuffle mean switch_band_correct_rate: `0.502045`
- blockshuffle mean peak_delay_lambda: `151.636364`
- best main strategy vs blockshuffle (delta_switch): `score_gating`

### Main vs Blockshuffle
| config_name | lambda_strategy | delta_align_vs_blockshuffle | delta_switch_vs_blockshuffle | delta_peakdelay_vs_blockshuffle | pass_core_checks_v3 |
| --- | --- | --- | --- | --- | --- |
| score_equal | score_equal | 0.0625443780391759 | -0.0320454545454546 | 103.63636363636363 | False |
| score_gating | score_gating | 0.13536304058905302 | 0.19295454545454538 | 103.63636363636363 | False |
| score_regime | score_regime | 0.13536304058905302 | 0.19295454545454538 | 103.63636363636363 | False |
