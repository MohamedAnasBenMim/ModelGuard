from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .features import FEATURE_NAMES, Transaction, transform
from .metrics import average_precision, confusion
from .model import FraudModel


@dataclass(slots=True)
class TrainingResult:
    model: FraudModel
    metrics: dict[str, float | dict[str, int]]


def synthetic_dataset(size: int = 2000, seed: int = 42) -> list[tuple[Transaction, int]]:
    randomizer = random.Random(seed)
    rows: list[tuple[Transaction, int]] = []
    for _ in range(size):
        transaction = Transaction(
            amount=round(randomizer.expovariate(1 / 180), 2),
            hour=randomizer.randrange(24),
            velocity_1h=randomizer.randrange(13),
            international=randomizer.random() < 0.18,
            email_age_days=randomizer.randrange(0, 1500),
        )
        values = transform(transaction)
        risk = (
            -5.8
            + 2.2 * values[0]
            + 3.6 * values[1]
            + 2.1 * values[2]
            + 2.6 * values[3]
            + 1.2 * values[4]
        )
        # Sharpen the synthetic decision boundary while retaining label noise.
        # This keeps the benchmark realistic enough to rank models without
        # making the generated labels perfectly deterministic.
        probability = 1 / (1 + math.exp(-2.5 * risk))
        rows.append((transaction, int(randomizer.random() < probability)))
    return rows


class TrainingPipeline:
    def __init__(self, learning_rate: float = 0.8, epochs: int = 2000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, rows: list[tuple[Transaction, int]]) -> FraudModel:
        weights = [0.0] * len(FEATURE_NAMES)
        bias = 0.0
        for _ in range(self.epochs):
            gradient = [0.0] * len(weights)
            bias_gradient = 0.0
            for transaction, label in rows:
                values = transform(transaction)
                score = sum(w * x for w, x in zip(weights, values, strict=True)) + bias
                probability = 1 / (1 + math.exp(-max(min(score, 30), -30)))
                error = probability - label
                for index, value in enumerate(values):
                    gradient[index] += error * value
                bias_gradient += error
            scale = self.learning_rate / len(rows)
            weights = [
                weight - scale * grad for weight, grad in zip(weights, gradient, strict=True)
            ]
            bias -= scale * bias_gradient
        fingerprint = hashlib.sha256(json.dumps(weights).encode()).hexdigest()[:8]
        return FraudModel(weights, bias, threshold=0.35, version=f"fraud-{fingerprint}")

    def train_and_evaluate(self, rows: list[tuple[Transaction, int]]) -> TrainingResult:
        split = int(len(rows) * 0.8)
        model = self.fit(rows[:split])
        labels = [label for _, label in rows[split:]]
        scores = [model.predict(transaction).fraud_probability for transaction, _ in rows[split:]]
        predictions = [score >= model.threshold for score in scores]
        return TrainingResult(
            model,
            {
                "pr_auc": round(average_precision(labels, scores), 4),
                "confusion": confusion(labels, predictions),
            },
        )


class ExperimentTracker:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, result: TrainingResult, parameters: dict[str, float | int]) -> None:
        history = json.loads(self.path.read_text()) if self.path.exists() else []
        history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": result.model.version,
                "parameters": parameters,
                "metrics": result.metrics,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
