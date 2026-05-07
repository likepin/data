# Traffic96 Stage3 Risk Windows

Scope: Stage3 closed-form eta2, with eta and target mask fixed before this diagnostic.

Key test observations:
- Overall test MSE gain vs Stage2 anchor: `0.0704%`.
- `gamma_floor` covers `90.54%` of test windows and contributes `103.94%` of total SSE gain.
- `gamma_active_gt_floor` covers `9.46%` of test windows but has MSE gain `-0.0343%`.
- `top_rank_5pct_gamma` has MSE gain `-0.0591%`.

Validation observation:
- Validation Fold 4 remains the strongest fold-level gain region: `0.5723%` MSE gain.

Interpretation:
- The current Stage3 eta2 result should not be framed as high-risk-window localization.
- Test improvement is mostly a weak global/floor correction effect; high-gamma active windows are still unstable.
- This supports keeping the guard narrative: lambda is useful as a diagnostic signal, but current dynamic correction is not yet a reliable high-risk attack module.

Files:
- `traffic96_stage3_eta2_risk_group_table.csv`: risk bucket metrics.
- `traffic96_stage3_eta2_fold_contribution.csv`: fold-level contribution metrics.
- `traffic96_stage3_eta2_top_risk_windows.csv`: best-gain and highest-gamma window lists.
- `traffic96_stage3_eta2_{val,test}_risk_windows_sample_stats.csv`: sample-level diagnostic table.
