## Phase A Block-Shuffle Summary

- blockshuffle mean directional_align_overall: `0.404256`
- blockshuffle mean switch_band_correct_rate: `0.442955`
- blockshuffle mean peak_delay_lambda: `125.181818`
- best main strategy vs blockshuffle (delta_switch): `score_equal`

### Main vs Blockshuffle
| config_name | lambda_strategy | delta_align_vs_blockshuffle | delta_switch_vs_blockshuffle | delta_peakdelay_vs_blockshuffle | pass_core_checks_v3 |
| --- | --- | --- | --- | --- | --- |
| score_equal | score_equal | 0.21168728362587713 | 0.3895454545454547 | 77.18181818181819 | False |
| score_gating | score_gating | 0.05751516169933246 | -0.007954545454545325 | 4.181818181818187 | False |
| score_regime | score_regime | 0.05751516169933246 | -0.007954545454545325 | 4.181818181818187 | False |
