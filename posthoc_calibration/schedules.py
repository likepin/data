from __future__ import annotations

import numpy as np


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def build_schedules(
    lambda_calib: np.ndarray,
    q_lows: list[float],
    q_highs: list[float],
    gamma_mins: list[float],
    gamma_maxs: list[float],
) -> list[dict]:
    rows = []
    for q_low in q_lows:
        for q_high in q_highs:
            if q_high <= q_low:
                continue
            low_value = float(np.quantile(lambda_calib, q_low))
            high_value = float(np.quantile(lambda_calib, q_high))
            if high_value <= low_value + 1e-12:
                continue
            for gamma_min in gamma_mins:
                for gamma_max in gamma_maxs:
                    if gamma_max <= gamma_min + 1e-12:
                        continue
                    rows.append(
                        {
                            "q_low": float(q_low),
                            "q_high": float(q_high),
                            "q_low_value": low_value,
                            "q_high_value": high_value,
                            "gamma_min": float(gamma_min),
                            "gamma_max": float(gamma_max),
                        }
                    )
    if not rows:
        raise ValueError("No valid q/gamma schedules were generated.")
    return rows


def build_active_ratio_schedules(
    active_ratios: list[float],
    gamma_mins: list[float],
    gamma_maxs: list[float],
) -> list[dict]:
    rows = []
    for ratio in active_ratios:
        ratio = float(ratio)
        if ratio <= 0.0 or ratio > 1.0:
            continue
        q_low = 1.0 - ratio
        q_high = 1.0
        for gamma_min in gamma_mins:
            for gamma_max in gamma_maxs:
                if gamma_max <= gamma_min + 1e-12:
                    continue
                rows.append(
                    {
                        "schedule_type": "active_ratio",
                        "active_ratio_target": ratio,
                        "q_low": float(q_low),
                        "q_high": float(q_high),
                        "q_low_value": float(q_low),
                        "q_high_value": float(q_high),
                        "gamma_min": float(gamma_min),
                        "gamma_max": float(gamma_max),
                    }
                )
    if not rows:
        raise ValueError("No valid active-ratio schedules were generated.")
    return rows


def gamma_from_schedule(lambda_values: np.ndarray, schedule: dict) -> np.ndarray:
    denom = float(schedule["q_high_value"]) - float(schedule["q_low_value"])
    if abs(denom) <= 1e-12:
        return np.full_like(lambda_values, fill_value=float(schedule["gamma_min"]), dtype=np.float32)
    weight = np.clip(
        (lambda_values - float(schedule["q_low_value"])) / denom,
        0.0,
        1.0,
    )
    gamma = float(schedule["gamma_min"]) + (float(schedule["gamma_max"]) - float(schedule["gamma_min"])) * weight
    return gamma.astype(np.float32)


def active_ratio_from_gamma(gamma: np.ndarray, gamma_floor: float, active_eps: float) -> float:
    return float(np.mean(gamma > (float(gamma_floor) + float(active_eps))))
