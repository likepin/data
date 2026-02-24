import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from step5pp_simulate_gated_graph import subset_masks
from step5_utils import load_lambda_and_mask, mean_over_mask


def test_step5pp_sanity_metrics():
    data_dir = "synthetic_step3_v2"
    lambda_t, valid_mask, _, _ = load_lambda_and_mask(data_dir, logs=[])
    high_mask, _, low_mask, _, _, _, _ = subset_masks(lambda_t, valid_mask, 0.90, 0.50, logs=[])

    mean_lambda_high = mean_over_mask(lambda_t, high_mask)
    mean_lambda_low = mean_over_mask(lambda_t, low_mask)

    assert mean_lambda_high >= mean_lambda_low
