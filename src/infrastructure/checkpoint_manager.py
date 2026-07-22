# src/infrastructure/checkpoint_manager.py
"""Centralized Checkpoint Manager module for saving and loading execution states.

Enables fault tolerance and pipeline recovery by persisting state dictionaries
as serialized JSON files.
"""

import os
import json
import datetime
from typing import Any, Dict


class CheckpointManager:
    """Manages saving, loading, and listing application checkpoints."""

    def __init__(self, checkpoints_dir: str = "checkpoints"):
        """Initialize the Checkpoint Manager with the specified checkpoints directory.

        Parameters
        ----------
        checkpoints_dir : str
            Directory path where checkpoints are saved. Defaults to 'checkpoints'.
        """
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _get_filepath(self, checkpoint_name: str) -> str:
        """Construct the file path for a checkpoint file.

        Parameters
        ----------
        checkpoint_name : str
            Identifier name of the checkpoint.

        Returns
        -------
        str
            Full file path for the checkpoint JSON file.
        """
        # Strip potential file extensions added accidentally
        clean_name = os.path.splitext(checkpoint_name)[0]
        return os.path.join(self.checkpoints_dir, f"{clean_name}.json")

    def save_checkpoint(self, checkpoint_name: str, state_data: Dict[str, Any]) -> str:
        """Serialize and save the application state to a JSON checkpoint file.

        Parameters
        ----------
        checkpoint_name : str
            Identifier name of the checkpoint.
        state_data : Dict[str, Any]
            State dictionary containing execution progress, indices, or metrics.

        Returns
        -------
        str
            The file path where the checkpoint was written.
        """
        file_path = self._get_filepath(checkpoint_name)
        envelope = {
            "checkpoint_name": checkpoint_name,
            "saved_at": datetime.datetime.now().isoformat(),
            "state": state_data,
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(envelope, f, indent=4)
        except OSError as e:
            raise OSError(f"Failed to write checkpoint file at {file_path}: {e}") from e

        return file_path

    def load_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Load and deserialize an existing checkpoint file.

        Parameters
        ----------
        checkpoint_name : str
            Identifier name of the checkpoint.

        Returns
        -------
        Dict[str, Any]
            The state dictionary retrieved from the checkpoint envelope.
        """
        file_path = self._get_filepath(checkpoint_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                envelope = json.load(f)
            return envelope.get("state", {})
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Failed to read/decode checkpoint file at {file_path}: {e}") from e

    def has_checkpoint(self, checkpoint_name: str) -> bool:
        """Verify whether a specific checkpoint file exists on disk.

        Parameters
        ----------
        checkpoint_name : str
            Identifier name of the checkpoint.

        Returns
        -------
        bool
            True if the checkpoint file exists, False otherwise.
        """
        return os.path.exists(self._get_filepath(checkpoint_name))


if __name__ == "__main__":
    # Exercise and Manual Validation entry point
    manager = CheckpointManager()
    test_checkpoint = "infra_diagnostic"
    test_state = {
        "completed_step": 3,
        "total_steps": 10,
        "active_dataset": "megascale_d",
        "processed_records": 15420,
    }

    print("Saving checkpoint...")
    written_path = manager.save_checkpoint(test_checkpoint, test_state)
    print(f"Checkpoint saved successfully to {written_path}")

    print("Deleting memory object...")
    test_state_cleared = {}

    print("Restoring state from checkpoint...")
    restored_state = manager.load_checkpoint(test_checkpoint)

    # Validate state restoration matches original state
    if restored_state == test_state:
        print("\n--- Manual Validation ---")
        print("Checkpoint restoration matches original state:")
        print(json.dumps(restored_state, indent=2))
        print("-------------------------")
    else:
        print("FAILED: Restored state does not match original.")
