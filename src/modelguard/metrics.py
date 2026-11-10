from __future__ import annotations


def average_precision(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have equal non-zero length")
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels, strict=True), reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def confusion(labels: list[int], predictions: list[bool]) -> dict[str, int]:
    result = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for label, prediction in zip(labels, predictions, strict=True):
        key = (
            "tp"
            if label and prediction
            else "tn"
            if not label and not prediction
            else "fp"
            if prediction
            else "fn"
        )
        result[key] += 1
    return result
