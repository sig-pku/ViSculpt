"""Select and verify the PyTorch backend used by the SAM 3 service."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

TORCH_VERSION = "2.13.0"
TORCHVISION_VERSION = "0.28.0"


@dataclass(frozen=True)
class TorchBackend:
    """One PyTorch build in ViSculpt's tested Windows backend matrix."""

    name: str
    index_url: str
    cuda_version: str | None
    minimum_driver: tuple[int, int] | None = None
    minimum_compute_capability: tuple[int, int] | None = None
    maximum_compute_capability: tuple[int, int] | None = None

    def supports(self, gpu: WindowsGpu) -> bool:
        if self.cuda_version is None:
            return True
        if self.minimum_driver is None:
            return False
        if _version_pair(gpu.driver_version) < self.minimum_driver:
            return False
        if (
            self.minimum_compute_capability is not None
            and gpu.compute_capability < self.minimum_compute_capability
        ):
            return False
        return not (
            self.maximum_compute_capability is not None
            and gpu.compute_capability > self.maximum_compute_capability
        )


@dataclass(frozen=True)
class WindowsGpu:
    """NVIDIA GPU properties available before PyTorch is installed."""

    index: int
    name: str
    compute_capability: tuple[int, int]
    driver_version: tuple[int, ...]

    @property
    def torch_architecture(self) -> str:
        major, minor = self.compute_capability
        return f"sm_{major}{minor}"

    @property
    def display_compute_capability(self) -> str:
        major, minor = self.compute_capability
        return f"{major}.{minor}"

    @property
    def display_driver_version(self) -> str:
        return ".".join(str(value) for value in self.driver_version)


@dataclass(frozen=True)
class BackendSelection:
    """Selected backend plus the hardware evidence used for the decision."""

    backend: TorchBackend
    gpu: WindowsGpu | None
    reason: str


@dataclass(frozen=True)
class TorchProbe:
    """Results of the four post-install checks requested by ViSculpt."""

    cuda_available: bool
    cuda_version: str | None
    device_name: str | None
    cuda_arch_list: tuple[str, ...]
    expected_architecture: str | None
    compute_capability_supported: bool | None

    def failures_for(self, backend: TorchBackend) -> tuple[str, ...]:
        failures: list[str] = []
        if backend.cuda_version is None:
            if self.cuda_available:
                failures.append("torch.cuda.is_available() returned True")
            if self.cuda_version is not None:
                failures.append(
                    "torch.version.cuda is not None for the CPU backend"
                )
            return tuple(failures)

        if not self.cuda_available:
            failures.append("torch.cuda.is_available() returned False")
        if self.cuda_version != backend.cuda_version:
            failures.append(
                "torch.version.cuda returned "
                f"{self.cuda_version!r}; expected {backend.cuda_version!r}"
            )
        if not self.device_name:
            failures.append("torch.cuda.get_device_name() returned no name")
        if self.compute_capability_supported is not True:
            failures.append(
                f"{self.expected_architecture} is not present in "
                "torch.cuda.get_arch_list()"
            )
        return tuple(failures)


CPU_BACKEND = TorchBackend(
    name="cpu",
    index_url="https://download.pytorch.org/whl/cpu",
    cuda_version=None,
)

# Keep this ordered from most preferred to least preferred. These combinations
# match the PyTorch 2.13 Windows x86-64 release matrix. CUDA 13 requires Turing
# or newer. CUDA 12.6 retains Maxwell through Hopper support.
WINDOWS_CUDA_BACKENDS = (
    TorchBackend(
        name="cu130",
        index_url="https://download.pytorch.org/whl/cu130",
        cuda_version="13.0",
        minimum_driver=(580, 0),
        minimum_compute_capability=(7, 5),
    ),
    TorchBackend(
        name="cu126",
        index_url="https://download.pytorch.org/whl/cu126",
        cuda_version="12.6",
        minimum_driver=(560, 76),
        minimum_compute_capability=(5, 0),
        maximum_compute_capability=(9, 9),
    ),
)


_TORCH_PROBE_PROGRAM = """
import json
import sys
import torch

expected_architecture = sys.argv[1] or None
cuda_available = bool(torch.cuda.is_available())
cuda_version = torch.version.cuda
device_name = torch.cuda.get_device_name() if cuda_available else None
cuda_arch_list = list(torch.cuda.get_arch_list())
compute_capability_supported = (
    expected_architecture in cuda_arch_list
    if expected_architecture is not None
    else None
)
print(json.dumps({
    "cuda_available": cuda_available,
    "cuda_version": cuda_version,
    "device_name": device_name,
    "cuda_arch_list": cuda_arch_list,
    "expected_architecture": expected_architecture,
    "compute_capability_supported": compute_capability_supported,
}))
""".strip()


def detect_windows_gpu(nvidia_smi: str | None = None) -> WindowsGpu | None:
    """Return CUDA device zero as reported by ``nvidia-smi`` when available."""

    executable = nvidia_smi or shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    rows = list(csv.reader(result.stdout.splitlines()))
    if not rows:
        return None
    gpus: list[WindowsGpu] = []
    for row in rows:
        try:
            index_text, name, capability_text, driver_text = (
                value.strip() for value in row
            )
            gpus.append(
                WindowsGpu(
                    index=int(index_text),
                    name=name,
                    compute_capability=_parse_version(
                        capability_text,
                        length=2,
                    ),
                    driver_version=_parse_version(driver_text),
                )
            )
        except (TypeError, ValueError):
            continue
    return min(gpus, key=lambda gpu: gpu.index) if gpus else None


def select_windows_backend(gpu: WindowsGpu | None) -> BackendSelection:
    """Choose the fastest compatible backend from the tested matrix."""

    if gpu is None:
        return BackendSelection(
            backend=CPU_BACKEND,
            gpu=None,
            reason="No queryable NVIDIA GPU was found.",
        )
    for backend in WINDOWS_CUDA_BACKENDS:
        if backend.supports(gpu):
            return BackendSelection(
                backend=backend,
                gpu=gpu,
                reason=(
                    f"{gpu.name} (compute capability "
                    f"{gpu.display_compute_capability}, driver "
                    f"{gpu.display_driver_version}) matches {backend.name}."
                ),
            )
    return BackendSelection(
        backend=CPU_BACKEND,
        gpu=gpu,
        reason=(
            f"{gpu.name} (compute capability "
            f"{gpu.display_compute_capability}, driver "
            f"{gpu.display_driver_version}) is outside the tested CUDA "
            "backend matrix."
        ),
    )


def torch_install_command(
    uv: str,
    python: Path,
    backend: TorchBackend,
) -> list[str]:
    """Build the isolated, exact-version PyTorch installation command."""

    return [
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--reinstall",
        "--no-deps",
        "--index-url",
        backend.index_url,
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
    ]


def probe_torch_installation(
    python: Path,
    gpu: WindowsGpu | None,
    *,
    env: dict[str, str] | None = None,
) -> TorchProbe:
    """Run only the four supported post-install PyTorch CUDA checks."""

    expected_architecture = gpu.torch_architecture if gpu is not None else ""
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                _TORCH_PROBE_PROGRAM,
                expected_architecture,
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise RuntimeError(f"PyTorch post-install checks failed: {detail}") from error
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("PyTorch post-install checks returned no result.")
    try:
        payload = json.loads(lines[-1])
        return TorchProbe(
            cuda_available=bool(payload["cuda_available"]),
            cuda_version=payload["cuda_version"],
            device_name=payload["device_name"],
            cuda_arch_list=tuple(payload["cuda_arch_list"]),
            expected_architecture=payload["expected_architecture"],
            compute_capability_supported=payload[
                "compute_capability_supported"
            ],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "PyTorch post-install checks returned an invalid result."
        ) from error


def format_probe(probe: TorchProbe) -> str:
    """Format the four checks for an actionable setup transcript."""

    if probe.expected_architecture is None:
        architecture_check = "N/A (CPU backend)"
    else:
        architecture_check = (
            f"{probe.expected_architecture} in "
            f"{list(probe.cuda_arch_list)}: "
            f"{probe.compute_capability_supported}"
        )
    return "\n".join(
        (
            "PyTorch post-install checks:",
            f"  torch.cuda.is_available(): {probe.cuda_available}",
            f"  torch.version.cuda: {probe.cuda_version}",
            f"  torch.cuda.get_device_name(): {probe.device_name}",
            (
                "  GPU Compute Capability in torch.cuda.get_arch_list(): "
                f"{architecture_check}"
            ),
        )
    )


def _parse_version(value: str, *, length: int | None = None) -> tuple[int, ...]:
    parts = tuple(int(part) for part in value.strip().split("."))
    if not parts or (length is not None and len(parts) != length):
        raise ValueError(f"Invalid version: {value!r}")
    return parts


def _version_pair(value: tuple[int, ...]) -> tuple[int, int]:
    padded = (*value, 0, 0)
    return padded[0], padded[1]
