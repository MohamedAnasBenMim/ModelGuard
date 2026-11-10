from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI

from .features import Transaction
from .model import FraudModel

app = FastAPI(title="ModelGuard", version="1.0.0")


def load_model() -> FraudModel:
    return FraudModel.load(Path(os.getenv("MODEL_PATH", "models/current.json")))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict")
def predict(transaction: Transaction) -> dict[str, object]:
    return asdict(load_model().predict(transaction))
