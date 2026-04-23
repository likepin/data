from __future__ import annotations

import pandas as pd


def classify_mode(active_ratio: float, active_cutoff: float, active_eps: float) -> str:
    if active_ratio <= active_eps:
        return "Bypass"
    if active_ratio <= active_cutoff:
        return "Selective"
    return "Active"


def resolve_mode_status(
    passed_selection: bool,
    active_ratio: float,
    active_cutoff: float,
    active_eps: float,
) -> tuple[str, str]:
    if not passed_selection:
        return "Bypass", "guard_reject"
    if active_ratio <= active_eps:
        return "Bypass", "zero_activation"
    if active_ratio <= active_cutoff:
        return "Selective", "selective_activation"
    return "Active", "broad_activation"


def select_schedule(
    val_grid: pd.DataFrame,
    guard_c: float,
    guard_beta: float,
    active_cutoff: float,
    active_eps: float,
) -> tuple[dict, pd.DataFrame]:
    selected = val_grid.copy()
    best_idx = selected["posthoc_mse"].idxmin()
    best_row = selected.loc[best_idx]
    mse_threshold = float(best_row["posthoc_mse"] + best_row["posthoc_mse_se"])
    selected["one_se_mse_threshold"] = mse_threshold
    selected["passes_one_se"] = selected["posthoc_mse"] <= mse_threshold + 1e-12

    baseline_available = selected["baseline_mae"].notna()
    selected["mae_guard_sigma_threshold"] = selected["static_mae"] + float(guard_c) * selected["static_mae_std"]
    selected["mae_guard_budget_threshold"] = (
        selected["static_mae"] + float(guard_beta) * (selected["baseline_mae"] - selected["static_mae"]).clip(lower=0.0)
    )
    selected.loc[~baseline_available, "mae_guard_budget_threshold"] = float("nan")
    selected["passes_mae_sigma_guard"] = selected["posthoc_mae"] <= selected["mae_guard_sigma_threshold"] + 1e-12
    selected["passes_mae_budget_guard"] = True
    selected.loc[baseline_available, "passes_mae_budget_guard"] = (
        selected.loc[baseline_available, "posthoc_mae"]
        <= selected.loc[baseline_available, "mae_guard_budget_threshold"] + 1e-12
    )
    selected["budget_guard_enabled"] = baseline_available
    selected["passes_mae_guard"] = selected["passes_mae_sigma_guard"] & selected["passes_mae_budget_guard"]
    selected["passes_selection"] = selected["passes_one_se"] & selected["passes_mae_guard"]
    mode_pairs = [
        resolve_mode_status(
            passed_selection=bool(passed),
            active_ratio=float(active),
            active_cutoff=active_cutoff,
            active_eps=active_eps,
        )
        for passed, active in zip(selected["passes_selection"], selected["active_ratio"])
    ]
    selected["mode_status"] = [pair[0] for pair in mode_pairs]
    selected["mode_reason"] = [pair[1] for pair in mode_pairs]
    selected["selected"] = False

    candidates = selected[selected["passes_selection"]].copy()
    if not candidates.empty:
        candidates = candidates.sort_values(["posthoc_mse", "posthoc_mae", "active_ratio"], ascending=[True, True, True])
        chosen = candidates.iloc[0].to_dict()
        chosen["selected"] = True
        chosen["selection_reason"] = "one_se_and_double_guard"
        selected.loc[candidates.index[0], "selected"] = True
        return chosen, selected

    fallback = selected.iloc[0].to_dict()
    fallback.update(
        {
            "q_low": 1.0,
            "q_high": 1.0,
            "q_low_value": float(selected["q_high_value"].max()),
            "q_high_value": float(selected["q_high_value"].max()),
            "gamma_min": 0.0,
            "gamma_max": 0.0,
            "gamma_mean": 0.0,
            "gamma_min_actual": 0.0,
            "gamma_max_actual": 0.0,
            "gamma_above_min_fraction": 0.0,
            "active_ratio": 0.0,
            "posthoc_mse": float(fallback["static_mse"]),
            "posthoc_mse_std": float(fallback["static_mse_std"]),
            "posthoc_mse_se": float(fallback["static_mse_se"]),
            "posthoc_mae": float(fallback["static_mae"]),
            "posthoc_mae_std": float(fallback["static_mae_std"]),
            "posthoc_mae_se": float(fallback["static_mae_se"]),
            "mse_gain_pct": 0.0,
            "mae_gain_pct": 0.0,
            "passes_one_se": False,
            "passes_mae_sigma_guard": False,
            "passes_mae_budget_guard": False,
            "passes_mae_guard": False,
            "passes_selection": False,
            "mode_status": "Bypass",
            "mode_reason": "guard_reject",
            "selected": True,
            "selection_reason": "fallback_static_only",
        }
    )
    return fallback, selected
