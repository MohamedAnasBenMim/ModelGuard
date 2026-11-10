from pathlib import Path

from modelguard.features import Transaction, transform
from modelguard.metrics import average_precision
from modelguard.model import FraudModel
from modelguard.monitoring import drift_report
from modelguard.pipeline import ExperimentTracker, TrainingPipeline, synthetic_dataset


def test_training_reaches_useful_pr_auc() -> None:
    result = TrainingPipeline(epochs=120).train_and_evaluate(synthetic_dataset(1200))
    assert result.metrics["pr_auc"] > 0.45


def test_model_round_trip_preserves_predictions(tmp_path: Path) -> None:
    model = TrainingPipeline(epochs=20).fit(synthetic_dataset(200))
    path = tmp_path / "model.json"
    model.save(path)
    transaction = Transaction(900, 2, 9, True, 1)
    assert FraudModel.load(path).predict(transaction) == model.predict(transaction)


def test_train_serving_features_are_consistent() -> None:
    model = TrainingPipeline(epochs=20).fit(synthetic_dataset(200))
    transaction = Transaction(200, 14, 2, False, 500)
    assert (
        abs(
            model.predict(transaction).fraud_probability
            - model.score_features(transform(transaction))
        )
        < 1e-6
    )


def test_drift_monitor_detects_shift() -> None:
    report = drift_report(
        [float(value) for value in range(100)], [float(value + 200) for value in range(100)]
    )
    assert report["drift_detected"] is True


def test_experiment_tracker_appends_runs(tmp_path: Path) -> None:
    result = TrainingPipeline(epochs=5).train_and_evaluate(synthetic_dataset(100))
    tracker = ExperimentTracker(tmp_path / "experiments.json")
    tracker.log(result, {"epochs": 5})
    tracker.log(result, {"epochs": 5})
    assert (tmp_path / "experiments.json").read_text().count(result.model.version) == 2


def test_average_precision_validates_inputs() -> None:
    assert average_precision([1, 0], [0.9, 0.1]) == 1.0
