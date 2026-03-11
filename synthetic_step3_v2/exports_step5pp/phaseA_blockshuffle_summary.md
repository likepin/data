## Phase A Block-Shuffle Summary

- blockshuffle mean directional_align_overall: `0.402556`
- blockshuffle mean switch_band_correct_rate: `0.404318`
- blockshuffle mean peak_delay_lambda: `114.636364`
- best main strategy vs blockshuffle (delta_switch): `score_gating`

### Main vs Blockshuffle
| config_name | lambda_strategy | delta_align_vs_blockshuffle | delta_switch_vs_blockshuffle | delta_peakdelay_vs_blockshuffle | pass_core_checks_v3 | pass_core_checks_v3_v2 |
| --- | --- | --- | --- | --- | --- | --- |
| score_equal | score_equal | 0.2323700249962496 | 0.3831818181818181 | 66.63636363636364 | False | False |
| score_gating | score_gating | 0.1404736850131919 | 0.4356818181818181 | -16.36363636363636 | False | False |
| score_regime | score_regime | 0.19328928841129722 | 0.4006818181818182 | 66.63636363636364 | True | True |
