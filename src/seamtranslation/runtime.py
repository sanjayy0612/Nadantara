from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    device: str
    torch_available: bool


def detect_runtime() -> RuntimeInfo:
    try:
        import torch
    except ImportError:
        return RuntimeInfo("cpu", False)
    if torch.backends.mps.is_available():
        return RuntimeInfo("mps", True)
    if torch.cuda.is_available():
        return RuntimeInfo("cuda", True)
    return RuntimeInfo("cpu", True)
