from __future__ import annotations

import math


def population_stability_index(
    reference: list[float], current: list[float], bins: int = 10
) -> float:
    if not reference or not current:
        raise ValueError("both populations are required")
    low, high = min(reference + current), max(reference + current)
    width = (high - low) / bins or 1.0

    def distribution(values: list[float]) -> list[float]:
        counts = [0] * bins
        for value in values:
            index = min(int((value - low) / width), bins - 1)
            counts[index] += 1
        return [max(count / len(values), 1e-6) for count in counts]

    expected, actual = distribution(reference), distribution(current)
    return sum((a - e) * math.log(a / e) for e, a in zip(expected, actual, strict=True))


def drift_report(
    reference: list[float], current: list[float], threshold: float = 0.2
) -> dict[str, float | bool]:
    psi = population_stability_index(reference, current)
    return {"psi": round(psi, 4), "drift_detected": psi >= threshold, "threshold": threshold}
