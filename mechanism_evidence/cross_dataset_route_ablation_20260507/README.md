# Cross-Dataset Route Ablation

Purpose:
- Freeze a route-level ablation table across ETTh1/Weather/ECL/Solar/Traffic at horizon 96, with Solar-192 as a cross-horizon extension.
- Keep train-time static residual, guarded post-hoc dynamic calibration, and prediction-level adaptive fusion separate.
- Prevent mixed claims such as treating prediction-level adaptive fusion gains as post-hoc dynamic gains.
- ETTh1 is frozen under the current `ParCorr ridgebase_sparse` backend rather than the older `parcorr_regen` interface.

Compact table:

| Dataset | Horizon | Baseline MSE/MAE | Static Anchor MSE/MAE | Guarded Post-Hoc Dynamic | Adaptive Fusion / Stage3 | Final Route |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 96 | 0.386865 / 0.404862 | 0.388214 / 0.406042; dBase -0.35% / -0.29% | Selective; 0.388088 / 0.405779; dStatic +0.03% / +0.06% | 0.380261 / 0.399500; dStatic +2.05% / +1.61%; Stage3 dAdaptive -0.43% / -0.09% | Adaptive fusion headline; Stage3 lambda/dynamic add-on is negative, and guarded post-hoc stays Selective but weaker than the fusion route. |
| Weather | 96 | 0.180918 / 0.221948 | 0.173706 / 0.214354; dBase +3.99% / +3.42% | Active_MSE_only; 0.173233 / 0.215034; dStatic +0.27% / -0.32% | n/a | Static anchor headline; post-hoc dynamic is MSE-positive but MAE-negative. |
| ECL | 96 | 0.148004 / 0.239849 | 0.144953 / 0.237570; dBase +2.06% / +0.95% | Bypass; 0.144953 / 0.237570; dStatic +0.00% / +0.00% | n/a | Static anchor headline; strict guarded dynamic branch bypasses. |
| Solar-96 | 96 | 0.203957 / 0.236751 | 0.205510 / 0.231356; dBase -0.76% / +2.28% | Selective; 0.204988 / 0.230618; dStatic +0.25% / +0.32% | 0.196010 / 0.227003; dStatic +4.62% / +1.88%; Stage3 dAdaptive +0.05% / +0.02%; MSE-primary dAdaptive +0.23% / +0.08% | Adaptive fusion headline; strict Stage3 is a weak add-on, while MSE-primary target gate is a secondary MSE-sensitive route. |
| Solar-192 | 192 | 0.240234 / 0.263654 | 0.243896 / 0.261298; dBase -1.52% / +0.89% | Selective; 0.243756 / 0.261294; dStatic +0.06% / +0.00% | 0.233232 / 0.254287; dStatic +4.37% / +2.68%; Stage3 dAdaptive +0.00% / +0.00%; MSE-primary dAdaptive +0.06% / +0.01% | Adaptive fusion headline; strict Stage3 falls back, while MSE-primary target gate is a small secondary MSE-sensitive route. |
| Traffic | 96 | 0.393648 / 0.269544 | 0.392304 / 0.268724; dBase +0.34% / +0.30% | Bypass; 0.392133 / 0.268711; dStatic +0.00% / +0.00% | 0.382640 / 0.259420; dStatic +2.46% / +3.46%; Stage3 dAdaptive +0.07% / +0.10% | Adaptive fusion headline; Stage3 lambda/dynamic is a weak add-on. |

Interpretation:
- `Static Anchor` is the stable backbone on Weather/ECL and a useful candidate family on Traffic, but it is not itself a positive standalone route on ETTh1 or Solar.
- `Post-hoc Dynamic` is selective rather than universal: ETTh1 and Solar can be Selective vs static, ECL bypasses, Weather is MSE-only, and Traffic's strict closed loop bypasses.
- `Adaptive Fusion` is now a prediction-level performance branch on Traffic, Solar, and ETTh1, but it should not be conflated with guarded post-hoc dynamic calibration.
- `MSE-primary` is a secondary dynamic route for MSE-sensitive settings; it is currently frozen for Solar-96/192 and is not a replacement for strict CACI.
- ETTh1 is no longer a hard negative overall once adaptive fusion is allowed, although its guarded post-hoc route stays below baseline and its Stage3 dynamic add-on is currently negative.
- Solar-96/192 both fall back under strict target-gated Stage3, while MSE-primary admits small loss-specific dynamic gains.
- This table should be used as method-route ablation rather than a simple component-toggle ablation.

Files:
- `cross_dataset_route_ablation_full.csv/md`: full numeric table.
- `cross_dataset_route_ablation_compact.csv/md`: paper-facing compact table.
- `cross_dataset_route_ablation_paper.csv/md`: most readable paper-facing table.
- `manifest.json`: source files and route patterns.
