"""The :class:`Device` record: one compute target, addressable by any backend.

A device is always identified by a *pair*: the runtime that exposes it and the
handle that runtime understands. ``cuda:0`` only means something to PyTorch;
``NPU`` only means something to OpenVINO; ``QNNExecutionProvider`` only means
something to ONNX Runtime. Collapsing them into a single string like ``"npu"``
is the mistake that forces a redesign the day a second NPU vendor appears, so
the pair is kept explicit everywhere and only flattened for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Runtime(str, Enum):
    """The software stack through which a device is reached."""

    TORCH_CUDA = "torch-cuda"
    TORCH_XPU = "torch-xpu"
    TORCH_MPS = "torch-mps"
    TORCH_CPU = "torch-cpu"
    TORCH_DIRECTML = "torch-directml"
    OPENVINO = "openvino"
    ONNXRUNTIME = "onnxruntime"
    REMOTE = "remote"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


class DeviceKind(str, Enum):
    """Physical class of the device. Drives default selection order."""

    GPU = "gpu"
    NPU = "npu"
    CPU = "cpu"
    REMOTE = "remote"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


class Vendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    QUALCOMM = "qualcomm"
    APPLE = "apple"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


#: Preference order used by ``--device auto``. A discrete GPU beats an NPU for
#: diffusion today (NPUs are power-efficient but slower on UNet-sized graphs and
#: constrained to static shapes); an NPU beats the CPU by a wide margin.
DEFAULT_KIND_PRIORITY: tuple[DeviceKind, ...] = (
    DeviceKind.GPU,
    DeviceKind.NPU,
    DeviceKind.CPU,
    DeviceKind.REMOTE,
)

#: Tie-break when several runtimes expose the same physical device. The same CPU
#: appears as both ``cpu`` (PyTorch) and ``ov:CPU`` (OpenVINO); ``--device cpu``
#: must land on the runtime whose backend is actually implemented, not on
#: whichever sorts first alphabetically.
RUNTIME_PREFERENCE: tuple[Runtime, ...] = (
    Runtime.TORCH_CUDA,
    Runtime.TORCH_XPU,
    Runtime.TORCH_MPS,
    Runtime.TORCH_CPU,
    Runtime.TORCH_DIRECTML,
    Runtime.OPENVINO,
    Runtime.ONNXRUNTIME,
    Runtime.REMOTE,
)


@dataclass(frozen=True, slots=True)
class Device:
    """One compute target.

    Attributes:
        runtime: Which stack exposes it (see :class:`Runtime`).
        handle: The identifier that runtime expects. ``"cuda:0"`` for PyTorch,
            ``"NPU"`` / ``"GPU.0"`` for OpenVINO, ``"QNNExecutionProvider"`` for
            ONNX Runtime.
        kind: GPU / NPU / CPU.
        vendor: Silicon vendor, when it can be determined.
        name: Human-readable description for the ``devices`` table.
        index: Ordinal within its runtime, for multi-GPU machines.
        total_memory_bytes: Dedicated memory, or ``None`` when the device shares
            system RAM (all NPUs, integrated GPUs). ``None`` means "unknown", not
            "zero" - the memory policy must treat it as such.
        compute_capability: ``(major, minor)`` on CUDA, ``None`` elsewhere.
        supports_dynamic_shapes: ``False`` for devices that must have the graph
            compiled for one fixed resolution (every NPU path today). The
            generator uses this to refuse or re-compile instead of producing
            garbage.
        supports_fp16 / supports_bf16: Numeric formats the device can actually
            execute, not merely accept.
        properties: Raw runtime-reported extras, shown by ``devices --verbose``.
        notes: Caveats worth telling the user (driver too old, provider present
            but untested for diffusion, ...).
    """

    runtime: Runtime
    handle: str
    kind: DeviceKind
    vendor: Vendor = Vendor.UNKNOWN
    name: str = ""
    index: int = 0
    total_memory_bytes: int | None = None
    compute_capability: tuple[int, int] | None = None
    supports_dynamic_shapes: bool = True
    supports_fp16: bool = False
    supports_bf16: bool = False
    #: Excluded from equality and hashing: a device is identified by its runtime
    #: and handle, not by whatever extras the runtime happened to report. Leaving
    #: a dict in the hash would also make the whole record unhashable, and
    #: PipelineSpec - which embeds a device plan - has to hash to be a cache key.
    properties: dict[str, Any] = field(default_factory=dict, compare=False)
    notes: tuple[str, ...] = field(default=(), compare=False)

    @property
    def id(self) -> str:
        """Stable selector, round-trippable through ``--device``.

        Example: ``torch-cuda:cuda:0``, ``openvino:NPU``,
        ``onnxruntime:QNNExecutionProvider``.
        """
        return f"{self.runtime.value}:{self.handle}"

    @property
    def short_id(self) -> str:
        """The shortest selector that is unambiguous in common setups."""
        if self.runtime is Runtime.TORCH_CUDA:
            return self.handle  # "cuda:0"
        if self.runtime is Runtime.TORCH_CPU:
            return "cpu"
        if self.runtime is Runtime.OPENVINO:
            return f"ov:{self.handle}"
        if self.runtime is Runtime.ONNXRUNTIME:
            return f"ort:{self.handle.removesuffix('ExecutionProvider')}"
        return self.id

    @property
    def total_memory_gb(self) -> float | None:
        if self.total_memory_bytes is None:
            return None
        return self.total_memory_bytes / (1024**3)

    @property
    def is_accelerator(self) -> bool:
        return self.kind in (DeviceKind.GPU, DeviceKind.NPU)

    def label(self) -> str:
        """One-line description for tables and logs."""
        mem = self.total_memory_gb
        mem_str = f", {mem:.1f} Go" if mem else ""
        return f"{self.name or self.handle} [{self.kind.value}{mem_str}] via {self.runtime.value}"

    def matches(self, selector: str) -> bool:
        """Whether a user-provided ``--device`` string designates this device.

        Accepted forms, from most to least specific::

            torch-cuda:cuda:0   full id
            cuda:0              torch handle
            ov:NPU / ort:QNN    short id
            cuda / npu / gpu    kind or runtime family
            auto                everything (handled by the caller)
        """
        sel = selector.strip().lower()
        if not sel:
            return False
        if sel in {self.id.lower(), self.short_id.lower(), self.handle.lower()}:
            return True
        if sel == self.kind.value:
            return True
        if sel == self.vendor.value:
            return True
        if sel == self.runtime.value:
            return True
        # "cuda" alone selects the first CUDA device; same for "xpu", "mps", "dml".
        family = {
            Runtime.TORCH_CUDA: "cuda",
            Runtime.TORCH_XPU: "xpu",
            Runtime.TORCH_MPS: "mps",
            Runtime.TORCH_DIRECTML: "dml",
            Runtime.OPENVINO: "openvino",
            Runtime.ONNXRUNTIME: "onnx",
        }.get(self.runtime)
        return sel == family
