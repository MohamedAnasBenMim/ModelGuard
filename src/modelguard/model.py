from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .features import FEATURE_NAMES, Transaction, transform


@dataclass(frozen=True, slots=True)
class Prediction:
    fraud_probability: float
    is_fraud: bool
    model_version: str


@dataclass(slots=True)
class FraudModel:
    weights: list[float]
    bias: float
    threshold: float = 0.5
    version: str = "unversioned"

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-max(min(value, 30), -30)))

    def score_features(self, values: list[float]) -> float:
        if len(values) != len(self.weights):
            raise ValueError("feature vector does not match model")
        return self._sigmoid(
            sum(w * x for w, x in zip(self.weights, values, strict=True)) + self.bias
        )

    def predict(self, transaction: Transaction) -> Prediction:
        probability = self.score_features(transform(transaction))
        return Prediction(round(probability, 6), probability >= self.threshold, self.version)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**asdict(self), "features": FEATURE_NAMES}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> FraudModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if tuple(payload.pop("features")) != FEATURE_NAMES:
            raise ValueError("model feature contract is incompatible")
        return cls(**payload)
