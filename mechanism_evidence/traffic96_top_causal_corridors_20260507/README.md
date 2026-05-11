# Traffic96 Top Causal Corridors Evidence

Purpose:
- Close the Traffic96 evidence around Stage2 variable alpha without adding new training or high-free-parameter search.
- Separate static-anchor reliance, graph centrality, local performance gain, and window-level lambda evidence.

Main readout:
- Stage2 test MSE gain vs best single anchor: `2.4370%`.
- Stage2 test MAE gain vs best single anchor: `3.3201%`.
- Top static-anchor corridor nodes by composite score: `567, 606, 103, 705, 597, 9, 454, 842, 551, 243`.
- Top stress/correction-energy nodes by energy+degree score: `567, 354, 746, 683, 253, 650, 472, 103, 542, 133`.
- Top pure correction-energy nodes: `840, 512, 642, 724, 545, 193, 817, 571, 73, 836`.
- Target 840 snapshot: `energy_share=13.69%, alpha=0.0414, test_mse_gain=7.4856%`.

Correlation diagnostics:
- Spearman(alpha_i, weighted_total_degree): `0.0510`.
- Spearman(alpha_i, per-target test MSE gain): `-0.4106`.
- Spearman(correction_energy, per-target test MSE gain): `-0.0906`.
- Top-5% overlap between alpha_i and weighted graph degree: `2/44 (4.55%)`.

Interpretation:
- If high-alpha nodes do not strongly overlap with high correction-energy nodes, this is not a failure: alpha_i measures static-anchor reliance, while correction energy measures baseline/static disagreement scale.
- Because lambda/gamma is window-level in the current protocol, this package does not claim variable-level risk response.
- The strongest defensible claim is that Stage2 obtains its main Traffic gain from variable-specific static-anchor allocation; Stage3 remains a weak positive add-on rather than a reliable high-risk-window attack.

Files:
- `traffic96_target_node_metrics.csv`: complete per-variable table.
- `traffic96_top_static_corridor_nodes.csv`: high alpha_i + graph degree + local gain ranking.
- `traffic96_top_stress_nodes.csv`: correction-energy + graph degree ranking.
- `traffic96_top_correction_energy_nodes.csv`: pure correction-energy ranking.
- `traffic96_top_alpha_nodes.csv`: pure alpha_i ranking.
- `traffic96_key_node_snapshot.csv`: union of top static, stress, correction-energy nodes plus Target 840.
- `traffic96_one_hop_edges_top_static_corridors.csv`: weighted one-hop edge list around top corridor nodes.
- `traffic96_corridor_overlap.csv`: top-set overlaps.
- `traffic96_corridor_correlations.csv`: Spearman diagnostics.
- `manifest.json`: source paths and summary metrics.
