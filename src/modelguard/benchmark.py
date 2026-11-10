from __future__ import annotations

import json
import time
from pathlib import Path

from .features import transform
from .monitoring import drift_report
from .pipeline import TrainingPipeline, synthetic_dataset


def benchmark() -> dict[str, object]:
    rows = synthetic_dataset()
    pipeline = TrainingPipeline()
    training_started = time.perf_counter()
    result = pipeline.train_and_evaluate(rows)
    training_ms = (time.perf_counter() - training_started) * 1000
    sample = rows[:500]
    latencies = []
    consistency = []
    for transaction, _ in sample:
        started = time.perf_counter()
        online = result.model.predict(transaction).fraud_probability
        latencies.append((time.perf_counter() - started) * 1000)
        batch = result.model.score_features(transform(transaction))
        consistency.append(abs(online - batch))
    reference = [transaction.amount for transaction, _ in rows[:500]]
    shifted = [value * 2.5 for value in reference]
    return {
        "pr_auc": result.metrics["pr_auc"],
        "training_ms": round(training_ms, 2),
        "serving_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
        "max_train_serving_delta": round(max(consistency), 8),
        "drift": drift_report(reference, shifted),
        "model_version": result.model.version,
    }


def main() -> None:
    report = benchmark()
    output = Path("benchmarks/results.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
