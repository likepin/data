| candidate_claim | evidence | verdict | paper_safe_framing |
| --- | --- | --- | --- |
| Solar-96 has stronger dynamic signal than Weather-96. | mean 0.032410 / val 6.917%; test 8.726% | support_with_guard | Solar shows clearer recoverable dynamic signal, but only under ideal scaling or heavy guard. |
| Solar-96 dynamic branch can be directly promoted to a positive active route. | mean -0.073537; positive-rate 0.704%; worst5 -0.359927 | reject_for_now | Current deployable gain-aware gates do not produce a positive active frontier. |
| The existing closed-loop schedule is useful but weak. | mean 0.000522; active-ratio 4.118% | support | Closed-loop scheduling contributes a tiny safe correction rather than a strong dynamic route. |
| Risk-return diagnostics justify bypass/guard behavior on Solar. | top decile mean -0.073537; bottom decile mean -0.434932 / mean -0.076210; nonzero-dynamic-rate 35.846% | support | Gain-aware probes identify safer windows but not reliable positive active corrections. |
| A probability gate alone is enough. | Target logistic AUC is useful, but top-k gain/CVaR remains non-positive. | reject | Solar reinforces the need for expected-gain and downside-risk audits. |
| Solar is a better next target than Traffic for refining dynamic gates. | mean 0.032410 and tractable 137-variable target-wise diagnostics. | support | Solar is the appropriate medium-scale case for dynamic-gate diagnostics before Traffic-scale deployment. |
