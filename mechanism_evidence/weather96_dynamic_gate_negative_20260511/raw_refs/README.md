# Raw Source References

Large diagnostic source directories are intentionally not copied in full.

- lambda adequacy: `C:\Users\cyl\Desktop\data\deltaA_signal_audit\weather96_static_pat3_lambda_adequacy`
- lambda gate probe: `C:\Users\cyl\Desktop\data\deltaA_signal_audit\weather96_static_pat3_lambda_gate_logistic_probe`

Rebuild commands:

```powershell
python lambda_adequacy_audit.py --profile weather96_static_pat3 --tag lambda_adequacy --closed-loop-tag full_guard_v2 --progress-every 1000
python lambda_gate_logistic_probe.py --profile weather96_static_pat3 --audit-tag lambda_adequacy --closed-loop-tag full_guard_v2 --tag lambda_gate_logistic_probe --progress-every 1000
python weather_dynamic_gate_negative_evidence_pack.py
```
