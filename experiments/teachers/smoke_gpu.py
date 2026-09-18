"""Print remote Colab runtime details without downloading a model."""

from __future__ import annotations

import platform
import sys


def _transformers_version() -> str:
    try:
        import transformers
    except ImportError:
        return "not-installed"
    return str(transformers.__version__)


def main() -> None:
    import torch

    print(f"python={platform.python_version()}")
    print(f"python_implementation={platform.python_implementation()}")
    print(f"python_full={sys.version.splitlines()[0]}")
    print(f"torch={torch.__version__}")
    print(f"transformers={_transformers_version()}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_version={torch.version.cuda}")

    if not torch.cuda.is_available():
        print("gpu_name=none")
        print("gpu_vram_gib=0.0")
        print("gpu_free_memory_gib=0.0")
        return

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    gib = 1024**3
    print(f"cuda_device_index={device}")
    print(f"gpu_name={properties.name}")
    print(f"gpu_vram_gib={properties.total_memory / gib:.3f}")
    print(f"gpu_free_memory_gib={free_bytes / gib:.3f}")
    print(f"gpu_total_memory_gib={total_bytes / gib:.3f}")


if __name__ == "__main__":
    main()
