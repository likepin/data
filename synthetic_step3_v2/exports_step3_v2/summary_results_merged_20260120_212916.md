## Summary (Merged: ParCorr vs CMIknn)

| prefix | mask | score_type | TP | FP | FN | Prec | Rec | F1 | SHD | K_true_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parcorr | TRUE-mask | magdiff | 2 | 4 | 4 | 0.333 | 0.333 | 0.333 | 8 | 6 |
| parcorr | TRUE-mask | signflip | 3 | 3 | 3 | 0.500 | 0.500 | 0.500 | 6 | 6 |
| parcorr | PRED-mask | magdiff | 2 | 4 | 4 | 0.333 | 0.333 | 0.333 | 8 | 6 |
| parcorr | PRED-mask | signflip | 2 | 4 | 4 | 0.333 | 0.333 | 0.333 | 8 | 6 |
| cmiknn | TRUE-mask | magdiff | 4 | 2 | 2 | 0.667 | 0.667 | 0.667 | 4 | 6 |
| cmiknn | TRUE-mask | signflip | 3 | 3 | 3 | 0.500 | 0.500 | 0.500 | 6 | 6 |
| cmiknn | PRED-mask | magdiff | 5 | 1 | 1 | 0.833 | 0.833 | 0.833 | 2 | 6 |
| cmiknn | PRED-mask | signflip | 1 | 5 | 5 | 0.167 | 0.167 | 0.167 | 10 | 6 |
