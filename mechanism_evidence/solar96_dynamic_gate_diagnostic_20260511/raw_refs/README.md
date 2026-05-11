# Raw Source References

Large diagnostic source directories are intentionally not copied in full.

- lambda adequacy: `C:\Users\cyl\Desktop\data\deltaA_signal_audit\solar96_static_lambda_adequacy`
- lambda gate probe: `C:\Users\cyl\Desktop\data\deltaA_signal_audit\solar96_static_lambda_gate_logistic_probe`

Rebuild commands:

```powershell
python lambda_adequacy_audit.py --profile solar96_static --tag lambda_adequacy --closed-loop-tag= --adaptive-alpha-csv deltaA_signal_audit\solar96_existing_prediction_ensemble\solar96_static_adaptive_alpha_variable_alpha.csv --progress-every 1000
python lambda_gate_logistic_probe.py --profile solar96_static --audit-tag lambda_adequacy --closed-loop-tag= --tag lambda_gate_logistic_probe --adaptive-alpha-csv deltaA_signal_audit\solar96_existing_prediction_ensemble\solar96_static_adaptive_alpha_variable_alpha.csv --progress-every 1000
python solar_dynamic_gate_diagnostic_evidence_pack.py
```
