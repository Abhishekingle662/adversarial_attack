import platform
import sys
import subprocess

import torch
import torchvision
import torchaudio
import transformers
import torchattacks
import numpy as np
import sklearn
import matplotlib
import tqdm

def print_header(title):
    print(f"\n=== {title} ===")

def main():
    # System information
    print_header("System Info")
    print(f"Platform:            {platform.platform()}")
    print(f"Python Version:      {platform.python_version()}")
    print(f"Python Executable:   {sys.executable}")

    # CUDA / cuDNN / Driver info
    print_header("CUDA / cuDNN / Driver Info")
    print(f"PyTorch CUDA version:     {torch.version.cuda}")
    print(f"CUDA available:           {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count:        {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"cuDNN version:            {torch.backends.cudnn.version()}")

    # nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        print("\nnvidia-smi info:")
        print(result.stdout.strip())
    except Exception as e:
        print("nvidia-smi not available or failed:", e)

    # Module versions
    print_header("Module Versions")
    versions = {
        "torch":            torch.__version__,
        "torchvision":      torchvision.__version__,
        "torchaudio":       torchaudio.__version__,
        "transformers":     transformers.__version__,
        "torchattacks":     torchattacks.__version__,
        "numpy":            np.__version__,
        "scikit-learn":     sklearn.__version__,
        "matplotlib":       matplotlib.__version__,
        "tqdm":             tqdm.__version__,
    }
    for name, ver in versions.items():
        print(f"{name:15}: {ver}")

if __name__ == "__main__":
    main()
