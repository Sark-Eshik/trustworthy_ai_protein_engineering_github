# tests/infrastructure/test_experiment_tracker.py
"""Unit tests for the experiment tracking module."""

import os
import json
import pytest
from src.infrastructure.experiment_tracker import ExperimentTracker


def test_experiment_tracker_records_runs(tmp_path):
    """Test serializing parameters and metrics into JSON artifacts."""
    tracker_dir = str(tmp_path / "tracker")
    tracker = ExperimentTracker(tracker_dir=tracker_dir)

    run_id = "test_unit_run"
    params = {"lr": 0.01, "epochs": 5}
    metrics = {"accuracy": 0.95}

    metadata = tracker.record_run(
        run_id=run_id,
        parameters=params,
        metrics=metrics,
        notes="Testing tracking logic.",
    )

    assert metadata["run_id"] == run_id
    assert metadata["parameters"] == params
    assert metadata["metrics"] == metrics

    # Verify storage serialization
    run_file = os.path.join(tracker_dir, f"run_{run_id}.json")
    assert os.path.exists(run_file)

    with open(run_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["run_id"] == run_id
    assert loaded["parameters"]["lr"] == 0.01
    assert loaded["metrics"]["accuracy"] == 0.95
