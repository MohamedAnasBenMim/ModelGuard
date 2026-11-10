from __future__ import annotations

import math
from dataclasses import dataclass

FEATURE_NAMES = ("amount", "velocity", "international", "email_risk", "night")


@dataclass(frozen=True, slots=True)
class Transaction:
    amount: float
    hour: int
    velocity_1h: int
    international: bool
    email_age_days: int


def transform(transaction: Transaction) -> list[float]:
    if transaction.amount < 0 or not 0 <= transaction.hour <= 23:
        raise ValueError("invalid transaction values")
    return [
        min(math.log1p(transaction.amount) / 10, 1.0),
        min(transaction.velocity_1h / 12, 1.0),
        float(transaction.international),
        1.0 / (1.0 + transaction.email_age_days / 30),
        float(transaction.hour < 6 or transaction.hour > 22),
    ]
