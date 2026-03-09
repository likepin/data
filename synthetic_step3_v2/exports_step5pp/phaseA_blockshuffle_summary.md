## Phase A Block-Shuffle Summary

- blockshuffle mean directional_align_overall: `0.404013`
- blockshuffle mean switch_band_correct_rate: `0.403864`
- blockshuffle mean peak_delay_lambda: `110.000000`
- best main strategy vs blockshuffle (delta_switch): `score_gating`

### Main vs Blockshuffle
| config_name | lambda_strategy | delta_align_vs_blockshuffle | delta_switch_vs_blockshuffle | delta_peakdelay_vs_blockshuffle | pass_core_checks_v3 | pass_core_checks_v3_v2 |
| --- | --- | --- | --- | --- | --- | --- |
| score_equal | score_equal | 0.057757793712010064 | 0.031136363636363573 | -11.0 | False | False |
| score_gating | score_gating | 0.1417913856366378 | 0.44613636363636355 | 56.0 | False | True |
| score_regime | score_regime | 0.14043297435391894 | 0.42613636363636354 | -11.0 | False | True |
