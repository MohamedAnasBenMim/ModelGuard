from pathlib import Path

from .pipeline import ExperimentTracker, TrainingPipeline, synthetic_dataset


def train() -> None:
    pipeline = TrainingPipeline()
    result = pipeline.train_and_evaluate(synthetic_dataset())
    result.model.save(Path("models/current.json"))
    ExperimentTracker(Path("models/experiments.json")).log(
        result, {"learning_rate": pipeline.learning_rate, "epochs": pipeline.epochs}
    )
    print(f"saved {result.model.version}: {result.metrics}")
