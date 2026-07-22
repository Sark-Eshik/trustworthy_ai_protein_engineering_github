# tests/infrastructure/test_checkpoint_manager.py
"""Unit tests for the application Checkpoint Manager module."""

import os
import pytest
from src.infrastructure.checkpoint_manager import CheckpointManager


def test_checkpoint_manager_save_and_load(tmp_path):
    """Test state serialization and identical restoration."""
    checkpoints_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(checkpoints_dir=checkpoints_dir)

    checkpoint_name = "test_run"
    state = {"epoch": 4, "loss": 0.045}

    path = manager.save_checkpoint(checkpoint_name, state)
    assert os.path.exists(path)

    # Restore and verify matching payload
    restored = manager.load_checkpoint(checkpoint_name)
    assert restored == state


def test_checkpoint_manager_missing_handling(tmp_path):
    """Test exception raising for nonexistent checkpoints."""
    manager = CheckpointManager(checkpoints_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        manager.load_checkpoint("nonexistent")
