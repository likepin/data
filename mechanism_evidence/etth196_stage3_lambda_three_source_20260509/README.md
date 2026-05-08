# ETTh1-96 Stage3 Lambda Three-Source Evidence

Boundary:
- Stage3 here means a lambda-gated dynamic increment added on top of the ETTh1 adaptive-alpha anchor.
- This should be interpreted as a negative audit over the dynamic branch, not as a new main performance route.

Key results:
- Adaptive-alpha anchor: `0.380261 / 0.399500`.
- `static_p0_dynamic`: selected `stage3_closed_form_all`, test `0.381877 / 0.399841`, gain vs adaptive anchor `-0.4250% / -0.0855%`.
- `static_mean_dynamic`: selected `stage3_closed_form_all`, test `0.381859 / 0.399825`, gain vs adaptive anchor `-0.4201% / -0.0816%`.
- Both selected rows saturate at `eta_mult=2.0` with `eta_raw=8.970` and `8.904` respectively, and both choose `target_mask=all`.

Interpretation:
- ETTh1 adaptive fusion remains the strongest current route.
- Adding the Traffic-style lambda-gated dynamic increment improves validation but hurts test under both dynamic-source choices.
- Therefore ETTh1 should currently be frozen as `adaptive fusion positive, Stage3 dynamic add-on negative`.

Files:
- `etth196_stage3_lambda_three_source_frozen_table.csv/md`: frozen comparison table.
- `raw_outputs/`: copied selected summaries, eta candidates, val grid, fold grid, and shuffle summaries.
- `manifest.json`: source run directories and copied raw output paths.
