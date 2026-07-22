# src/infrastructure/hardware_detection.py
"""Centralized Hardware Detection utility to profile system resources.

Queries CPU counts, RAM capacity, and GPU architectures to configure model
executors and execution context details.
"""

import os
import psutil
from typing import Any, Dict


def detect_hardware() -> Dict[str, Any]:
    """Profile the system to detect CPU, RAM, and GPU status.

    Returns
    -------
    Dict[str, Any]
        Profile dictionary containing hardware metrics and availability status.
    """
    # 1. Profile CPU
    # logical=False would give physical cores, logical=True gives logical cores (hyperthreaded)
    cpu_count_logical = os.cpu_count() or 1
    try:
        cpu_count_physical = psutil.cpu_count(logical=False) or cpu_count_logical
    except Exception:
        cpu_count_physical = cpu_count_logical

    # 2. Profile RAM
    try:
        mem_info = psutil.virtual_memory()
        total_ram_gb = round(mem_info.total / (1024**3), 2)
        available_ram_gb = round(mem_info.available / (1024**3), 2)
    except Exception:
        total_ram_gb = 0.0
        available_ram_gb = 0.0

    # 3. Profile GPU (PyTorch context integration)
    gpu_available = False
    gpu_count = 0
    gpu_name = "N/A"
    gpu_memory_gb = 0.0

    try:
        import torch

        # Handle CUDA (NVIDIA) or MPS (Apple Silicon Metal Performance Shaders)
        if torch.cuda.is_available():
            gpu_available = True
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            # Fetch total memory from the first device
            gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = round(gpu_memory_bytes / (1024**3), 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_available = True
            gpu_count = 1
            gpu_name = "Apple Silicon MPS"
            # MPS shares unified system memory dynamically; no static VRAM limit exists
            gpu_memory_gb = total_ram_gb
    except ImportError:
        pass  # PyTorch not installed or accessible

    return {
        "cpu_count_logical": cpu_count_logical,
        "cpu_count_physical": cpu_count_physical,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
    }


if __name__ == "__main__":
    # Exercise and Manual Validation entry point
    print("Detecting hardware resources...")
    profile = detect_hardware()

    # Create directories if missing
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "hardware_profile.md")

    # Generate certification report
    report_content = f"""# Hardware Profile Report

Generated on: {psutil.datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## CPU Details
- **Logical CPU Cores**: {profile['cpu_count_logical']}
- **Physical CPU Cores**: {profile['cpu_count_physical']}

## RAM Details
- **Total System RAM**: {profile['total_ram_gb']} GB
- **Available RAM**: {profile['available_ram_gb']} GB

## GPU Acceleration Details
- **GPU Available**: {profile['gpu_available']}
- **GPU Count**: {profile['gpu_count']}
- **GPU Name**: {profile['gpu_name']}
- **GPU VRAM / Dynamic Memory**: {profile['gpu_memory_gb']} GB
"""

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except OSError as e:
        print(f"ERROR: Failed to write hardware report: {e}")

    print("\n--- Manual Validation ---")
    print(report_content.strip())
    print("-------------------------")
