"""
Print (and return) everything about the machine that matters for training:
PyTorch / CUDA versions, the GPU, and whether this PyTorch build supports it.

    python -m neuralp.check_environment
"""

import json
import platform
import subprocess
import sys

import torch


def nvidia_smi_summary() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "nvidia-smi not available"


def environment_report() -> dict:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi": nvidia_smi_summary(),
        "warnings": [],
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        capability = f"sm_{props.major}{props.minor}"
        report.update({
            "gpu_name": props.name,
            "gpu_memory_gb": round(props.total_memory / 1e9, 1),
            "gpu_compute_capability": capability,
            "gpus_visible": torch.cuda.device_count(),
            "torch_compiled_for": torch.cuda.get_arch_list(),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
        if capability not in report["torch_compiled_for"]:
            report["warnings"].append(
                f"This PyTorch build has no kernels for {capability} (Blackwell needs a build for CUDA 12.8+, "
                "e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128).")
        try:  # a tiny 3D convolution to be sure the GPU really works
            x = torch.randn(1, 4, 16, 16, 16, device="cuda")
            torch.nn.functional.conv3d(x, torch.randn(8, 4, 3, 3, 3, device="cuda"), padding=1)
            torch.cuda.synchronize()
            report["conv3d_test"] = "ok"
        except Exception as error:  # noqa: BLE001 - report any failure
            report["conv3d_test"] = f"FAILED: {error}"
            report["warnings"].append("A 3D convolution failed on the GPU.")
    else:
        report["warnings"].append("No CUDA GPU visible: the benchmark would run on the CPU.")
    return report


if __name__ == "__main__":
    print(json.dumps(environment_report(), indent=2))
