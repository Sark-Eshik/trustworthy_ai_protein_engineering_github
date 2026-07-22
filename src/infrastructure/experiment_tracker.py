# src/infrastructure/experiment_tracker.py
"""Centralized experiment tracker module for logging and tracking model runs.

Records parameters, version data, metrics, system performance, and Git context
(if available) into structured run metadata.
"""

import datetime
import json
import os
import subprocess
from typing import Any, Dict, Optional


class ExperimentTracker:
    """Manages experiment runs, parameters, metrics, and metadata tracking."""

    def __init__(self, tracker_dir: str = "results/experiment_tracker"):
        """Initialize the Experiment Tracker with the specified tracking directory.

        Parameters
        ----------
        tracker_dir : str
            Directory path where metadata and tracking files are stored.
        """
        self.tracker_dir = tracker_dir
        os.makedirs(self.tracker_dir, exist_ok=True)

    def _get_git_commit_hash(self) -> str:
        """Fetch the current git commit hash if running in a git repository.

        Returns
        -------
        str
            The git commit hash, or 'N/A' if not a git repository or command fails.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return "N/A"

    def record_run(
        self,
        run_id: str,
        parameters: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record and serialize an experiment run's metadata to a JSON file.

        Parameters
        ----------
        run_id : str
            Unique identifier for the experiment run.
        parameters : Dict[str, Any]
            Configuration parameters or hyperparameters applied in this run.
        metrics : Optional[Dict[str, Any]]
            Experimental results or metrics achieved. Defaults to empty dict.
        version : str
            Software release or pipeline version tag. Defaults to '1.0.0'.
        notes : Optional[str]
            Optional annotations or qualitative remarks about this run.

        Returns
        -------
        Dict[str, Any]
            The generated metadata record dict.
        """
        metrics = metrics or {}
        git_hash = self._get_git_commit_hash()
        timestamp = datetime.datetime.now().isoformat()

        metadata: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": timestamp,
            "version": version,
            "git_hash": git_hash,
            "parameters": parameters,
            "metrics": metrics,
            "notes": notes,
        }

        run_file = os.path.join(self.tracker_dir, f"run_{run_id}.json")
        try:
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except OSError as e:
            # Fallback output in case of disk permission issues
            print(f"ERROR: Failed to write experiment run metadata to {run_file}: {e}")

        # Maintain or update a simple global index log of runs
        index_file = os.path.join(self.tracker_dir, "run_index.jsonl")
        try:
            with open(index_file, "a", encoding="utf-8") as f:
                index_entry = {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "version": version,
                    "git_hash": git_hash,
                }
                f.write(json.dumps(index_entry) + "\n")
        except OSError:
            pass  # Do not block the run recording if the central index file fails

        return metadata


if __name__ == "__main__":
    # Exercise and Manual Validation entry point
    tracker = ExperimentTracker()
    test_run_id = f"test_{int(datetime.datetime.now().timestamp())}"
    test_params = {"learning_rate": 0.001, "sparsity_threshold": 0.5, "epochs": 10}
    test_metrics = {"loss": 0.124, "accuracy": 0.941, "combined_sparsity_mean": 0.672}

    print("Registering experiment run...")
    recorded_metadata = tracker.record_run(
        run_id=test_run_id,
        parameters=test_params,
        metrics=test_metrics,
        notes="Infrastructure diagnostic execution.",
    )

    metadata_path = os.path.join("results", "experiment_tracker", f"run_{test_run_id}.json")
    if os.path.exists(metadata_path):
        print("\n--- Manual Validation ---")
        print(f"Run file successfully created at: {metadata_path}")
        print("Run Metadata content:")
        print(json.dumps(recorded_metadata, indent=2))
        print("-------------------------")
    else:
        print("FAILED: Run metadata file was not created.")
