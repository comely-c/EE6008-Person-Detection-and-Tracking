"""
Step 2 sanity check: confirm CUDA/GPU is visible to PyTorch.

Run:
    python scripts/check_gpu.py
"""

import torch


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name:  {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No CUDA device detected — CPU-only. "
              "Model inference will be slow or infeasible for the larger models.")


if __name__ == "__main__":
    main()
