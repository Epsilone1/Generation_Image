"""Runtime probes: enumerate every reachable accelerator, crash-free.

Each probe is independent and fully guarded. A missing package, a broken driver
or an OpenVINO plugin that throws on import must degrade to "this runtime is not
available", never to an exception escaping :func:`detect_devices` - otherwise
installing an optional NPU package could break generation on a working GPU.

The probes are also the *only* place in the codebase that imports torch,
openvino or onnxruntime for discovery purposes. Backends receive ready-made
:class:`Device` records.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from ..errors import DeviceNotFoundError
from .device import (
    DEFAULT_KIND_PRIORITY,
    RUNTIME_PREFERENCE,
    Device,
    DeviceKind,
    Runtime,
    Vendor,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Whether a runtime is installed and usable, for ``imagegen doctor``."""

    runtime: Runtime
    installed: bool
    version: str | None = None
    device_count: int = 0
    error: str | None = None
    install_hint: str | None = None


@dataclass(frozen=True, slots=True)
class DetectionReport:
    devices: tuple[Device, ...]
    statuses: tuple[RuntimeStatus, ...]


_CACHE: DetectionReport | None = None


def clear_detection_cache() -> None:
    """Forget the cached probe results (used by tests)."""
    global _CACHE
    _CACHE = None


# --------------------------------------------------------------------------- #
# Vendor / kind mapping helpers
# --------------------------------------------------------------------------- #

def _vendor_from_name(name: str) -> Vendor:
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "rtx" in lowered or "quadro" in lowered:
        return Vendor.NVIDIA
    if "amd" in lowered or "radeon" in lowered or "ryzen" in lowered:
        return Vendor.AMD
    if "intel" in lowered or "arc(tm)" in lowered or "iris" in lowered:
        return Vendor.INTEL
    if "qualcomm" in lowered or "snapdragon" in lowered or "adreno" in lowered or "hexagon" in lowered:
        return Vendor.QUALCOMM
    if "apple" in lowered:
        return Vendor.APPLE
    return Vendor.UNKNOWN


#: ONNX Runtime execution providers we know how to classify. Anything not listed
#: is still reported, as an unknown-kind device, rather than silently dropped.
_ORT_PROVIDERS: dict[str, tuple[DeviceKind, Vendor, bool, str]] = {
    # provider: (kind, vendor, supports_dynamic_shapes, note)
    "QNNExecutionProvider": (
        DeviceKind.NPU,
        Vendor.QUALCOMM,
        False,
        "NPU Hexagon: graphes a formes fixes, modeles pre-compiles (contextes QNN) requis.",
    ),
    "VitisAIExecutionProvider": (
        DeviceKind.NPU,
        Vendor.AMD,
        False,
        "NPU XDNA (Ryzen AI): necessite le Ryzen AI SDK et un modele quantifie.",
    ),
    "OpenVINOExecutionProvider": (
        DeviceKind.NPU,
        Vendor.INTEL,
        False,
        "EP OpenVINO: le peripherique reel (CPU/GPU/NPU) se choisit via device_type.",
    ),
    "DmlExecutionProvider": (
        DeviceKind.GPU,
        Vendor.UNKNOWN,
        True,
        "DirectML: tout GPU DX12 (AMD/Intel/NVIDIA/Adreno).",
    ),
    "CUDAExecutionProvider": (DeviceKind.GPU, Vendor.NVIDIA, True, ""),
    "TensorrtExecutionProvider": (DeviceKind.GPU, Vendor.NVIDIA, False, "TensorRT: moteur compile par forme."),
    "ROCMExecutionProvider": (DeviceKind.GPU, Vendor.AMD, True, ""),
    "MIGraphXExecutionProvider": (DeviceKind.GPU, Vendor.AMD, False, ""),
    "CoreMLExecutionProvider": (DeviceKind.NPU, Vendor.APPLE, False, "Apple Neural Engine."),
    "NvTensorRtRtxExecutionProvider": (
        DeviceKind.GPU,
        Vendor.NVIDIA,
        False,
        "TensorRT-RTX (Windows ML): moteur compile a la volee.",
    ),
    "WebGpuExecutionProvider": (DeviceKind.GPU, Vendor.UNKNOWN, True, ""),
}

#: Providers that are not compute devices and must not appear in the device list.
#: ``CPUExecutionProvider`` is already covered by the torch CPU entry;
#: ``AzureExecutionProvider`` routes to a cloud endpoint and is always "available".
_ORT_NON_DEVICE_PROVIDERS = frozenset({"CPUExecutionProvider", "AzureExecutionProvider"})

#: OpenVINO device prefixes -> (kind, note). OpenVINO names devices "CPU",
#: "GPU", "GPU.0", "GPU.1", "NPU".
_OV_KINDS: dict[str, DeviceKind] = {
    "CPU": DeviceKind.CPU,
    "GPU": DeviceKind.GPU,
    "NPU": DeviceKind.NPU,
}


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #

def _probe_torch() -> tuple[list[Device], list[RuntimeStatus]]:
    """PyTorch: CUDA, XPU (Intel), MPS (Apple) and the always-present CPU."""
    devices: list[Device] = []
    statuses: list[RuntimeStatus] = []

    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dep in practice
        hint = (
            "Installez PyTorch depuis l'index correspondant a votre materiel, ex. : "
            "uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision"
        )
        for runtime in (Runtime.TORCH_CUDA, Runtime.TORCH_CPU):
            statuses.append(
                RuntimeStatus(runtime, installed=False, error=str(exc), install_hint=hint)
            )
        return devices, statuses

    version = torch.__version__

    # --- CUDA -------------------------------------------------------------- #
    cuda_error: str | None = None
    cuda_devices: list[Device] = []
    try:
        if torch.cuda.is_available():
            bf16_ok = bool(torch.cuda.is_bf16_supported())
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                capability = (props.major, props.minor)
                notes: list[str] = []
                # Blackwell (sm_120) needs a CUDA 12.8+ build; an older wheel
                # loads but every kernel launch fails with "no kernel image".
                if capability >= (12, 0) and (torch.version.cuda or "0") < "12.8":
                    notes.append(
                        f"GPU sm_{props.major}{props.minor} avec une roue CUDA {torch.version.cuda} : "
                        "reinstallez torch depuis l'index cu128 ou superieur."
                    )
                cuda_devices.append(
                    Device(
                        runtime=Runtime.TORCH_CUDA,
                        handle=f"cuda:{index}",
                        kind=DeviceKind.GPU,
                        vendor=Vendor.NVIDIA,
                        name=props.name,
                        index=index,
                        total_memory_bytes=props.total_memory,
                        compute_capability=capability,
                        supports_dynamic_shapes=True,
                        supports_fp16=True,
                        # bf16 needs Ampere (sm_80) or newer.
                        supports_bf16=bf16_ok and capability >= (8, 0),
                        properties={
                            "cuda_runtime": torch.version.cuda,
                            "multi_processor_count": props.multi_processor_count,
                        },
                        notes=tuple(notes),
                    )
                )
    except Exception as exc:
        cuda_error = str(exc)
        logger.debug("CUDA probe failed", exc_info=True)

    devices.extend(cuda_devices)
    statuses.append(
        RuntimeStatus(
            Runtime.TORCH_CUDA,
            installed=bool(cuda_devices),
            version=f"torch {version} / cuda {torch.version.cuda}",
            device_count=len(cuda_devices),
            error=cuda_error,
            install_hint=None
            if cuda_devices
            else "Aucun GPU NVIDIA visible. Verifiez le pilote, ou installez la roue CUDA : "
            "uv pip install --index-url https://download.pytorch.org/whl/cu128 torch",
        )
    )

    # --- XPU (Intel Arc / iGPU via PyTorch) -------------------------------- #
    xpu_devices: list[Device] = []
    xpu_error: str | None = None
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for index in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(index)
                name = getattr(props, "name", f"Intel XPU {index}")
                xpu_devices.append(
                    Device(
                        runtime=Runtime.TORCH_XPU,
                        handle=f"xpu:{index}",
                        kind=DeviceKind.GPU,
                        vendor=Vendor.INTEL,
                        name=name,
                        index=index,
                        total_memory_bytes=getattr(props, "total_memory", None),
                        supports_fp16=True,
                        supports_bf16=True,
                    )
                )
    except Exception as exc:
        xpu_error = str(exc)
    devices.extend(xpu_devices)
    if xpu_devices or xpu_error:
        statuses.append(
            RuntimeStatus(
                Runtime.TORCH_XPU,
                installed=bool(xpu_devices),
                version=version,
                device_count=len(xpu_devices),
                error=xpu_error,
            )
        )

    # --- MPS (Apple Silicon) ----------------------------------------------- #
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(
                Device(
                    runtime=Runtime.TORCH_MPS,
                    handle="mps",
                    kind=DeviceKind.GPU,
                    vendor=Vendor.APPLE,
                    name="Apple Silicon (Metal)",
                    supports_fp16=True,
                    supports_bf16=True,
                )
            )
            statuses.append(
                RuntimeStatus(Runtime.TORCH_MPS, installed=True, version=version, device_count=1)
            )
    except Exception:  # pragma: no cover - non-macOS
        pass

    # --- DirectML ----------------------------------------------------------- #
    dml_devices: list[Device] = []
    dml_error: str | None = None
    try:
        import torch_directml  # type: ignore[import-not-found]

        if torch_directml.is_available():
            for index in range(torch_directml.device_count()):
                name = torch_directml.device_name(index)
                dml_devices.append(
                    Device(
                        runtime=Runtime.TORCH_DIRECTML,
                        handle=f"privateuseone:{index}",
                        kind=DeviceKind.GPU,
                        vendor=_vendor_from_name(name),
                        name=f"{name} (DirectML)",
                        index=index,
                        supports_fp16=True,
                        notes=("DirectML: pas de bf16, pas de torch.compile.",),
                    )
                )
    except ImportError:
        dml_error = None
    except Exception as exc:
        dml_error = str(exc)
    devices.extend(dml_devices)
    statuses.append(
        RuntimeStatus(
            Runtime.TORCH_DIRECTML,
            installed=bool(dml_devices),
            device_count=len(dml_devices),
            error=dml_error,
            install_hint="uv pip install torch-directml  # GPU DX12 (AMD/Intel) via PyTorch",
        )
    )

    # --- CPU: always there, always last ------------------------------------ #
    devices.append(
        Device(
            runtime=Runtime.TORCH_CPU,
            handle="cpu",
            kind=DeviceKind.CPU,
            vendor=_vendor_from_name(_cpu_name()),
            name=_cpu_name(),
            total_memory_bytes=_system_memory_bytes(),
            supports_fp16=False,  # fp16 on CPU is emulated: slower than fp32.
            supports_bf16=_cpu_supports_bf16(torch),
            notes=("Tres lent pour la diffusion (plusieurs minutes par image).",),
        )
    )
    statuses.append(
        RuntimeStatus(Runtime.TORCH_CPU, installed=True, version=version, device_count=1)
    )
    return devices, statuses


def _probe_openvino() -> tuple[list[Device], list[RuntimeStatus]]:
    """OpenVINO: the supported path to Intel NPUs (Core Ultra "AI Boost")."""
    try:
        import openvino as ov
    except ImportError:
        return [], [
            RuntimeStatus(
                Runtime.OPENVINO,
                installed=False,
                install_hint='uv pip install "imagegen[openvino]"  # NPU/GPU Intel',
            )
        ]
    except Exception as exc:  # pragma: no cover - broken install
        return [], [RuntimeStatus(Runtime.OPENVINO, installed=False, error=str(exc))]

    devices: list[Device] = []
    error: str | None = None
    try:
        core = ov.Core()
        for handle in core.available_devices:
            base = handle.split(".")[0].upper()
            kind = _OV_KINDS.get(base, DeviceKind.CPU)
            name, capabilities = _ov_device_info(core, handle)
            vendor = _vendor_from_name(name)
            notes: list[str] = []
            if kind is DeviceKind.NPU:
                notes.append(
                    "NPU: le modele doit etre recompile pour une resolution fixe "
                    "(reshape statique) avant inference."
                )
            if kind is DeviceKind.GPU and vendor is not Vendor.INTEL:
                # The OpenVINO GPU plugin enumerates any OpenCL device, including
                # non-Intel GPUs it cannot actually accelerate. Reported, but flagged.
                notes.append(
                    "GPU non-Intel expose par le plugin OpenCL d'OpenVINO : chemin non supporte, "
                    "preferez le backend PyTorch pour ce GPU."
                )
            devices.append(
                Device(
                    runtime=Runtime.OPENVINO,
                    handle=handle,
                    kind=kind,
                    vendor=vendor if vendor is not Vendor.UNKNOWN else Vendor.INTEL,
                    name=name,
                    total_memory_bytes=_system_memory_bytes() if kind is DeviceKind.NPU else None,
                    supports_dynamic_shapes=kind is not DeviceKind.NPU,
                    supports_fp16="FP16" in capabilities,
                    supports_bf16="BF16" in capabilities,
                    properties={"capabilities": sorted(capabilities)},
                    notes=tuple(notes),
                )
            )
    except Exception as exc:
        error = str(exc)
        logger.debug("OpenVINO probe failed", exc_info=True)

    return devices, [
        RuntimeStatus(
            Runtime.OPENVINO,
            installed=True,
            version=getattr(ov, "__version__", None),
            device_count=len(devices),
            error=error,
        )
    ]


def _ov_device_info(core: object, handle: str) -> tuple[str, set[str]]:
    """Read FULL_DEVICE_NAME and OPTIMIZATION_CAPABILITIES, tolerating old APIs."""
    name = handle
    capabilities: set[str] = set()
    for prop, target in (("FULL_DEVICE_NAME", "name"), ("OPTIMIZATION_CAPABILITIES", "caps")):
        try:
            value = core.get_property(handle, prop)  # type: ignore[attr-defined]
        except Exception:
            continue
        if target == "name" and value:
            name = str(value)
        elif target == "caps" and value:
            capabilities = {str(item).upper() for item in value}
    return name, capabilities


def _probe_onnxruntime() -> tuple[list[Device], list[RuntimeStatus]]:
    """ONNX Runtime: the common substrate for Qualcomm/AMD NPUs and DirectML."""
    try:
        import onnxruntime as ort
    except ImportError:
        return [], [
            RuntimeStatus(
                Runtime.ONNXRUNTIME,
                installed=False,
                install_hint='uv pip install "imagegen[onnx-directml]"  # ou [onnx-qnn] sur Snapdragon',
            )
        ]
    except Exception as exc:  # pragma: no cover
        return [], [RuntimeStatus(Runtime.ONNXRUNTIME, installed=False, error=str(exc))]

    devices: list[Device] = []
    error: str | None = None
    try:
        for provider in ort.get_available_providers():
            if provider in _ORT_NON_DEVICE_PROVIDERS:
                continue
            kind, vendor, dynamic, note = _ORT_PROVIDERS.get(
                provider, (DeviceKind.GPU, Vendor.UNKNOWN, True, "Provider non reference.")
            )
            devices.append(
                Device(
                    runtime=Runtime.ONNXRUNTIME,
                    handle=provider,
                    kind=kind,
                    vendor=vendor,
                    name=provider.removesuffix("ExecutionProvider"),
                    total_memory_bytes=_system_memory_bytes() if kind is DeviceKind.NPU else None,
                    supports_dynamic_shapes=dynamic,
                    supports_fp16=True,
                    notes=(note,) if note else (),
                )
            )
    except Exception as exc:
        error = str(exc)
        logger.debug("onnxruntime probe failed", exc_info=True)

    return devices, [
        RuntimeStatus(
            Runtime.ONNXRUNTIME,
            installed=True,
            version=getattr(ort, "__version__", None),
            device_count=len(devices),
            error=error,
        )
    ]


# --------------------------------------------------------------------------- #
# System info helpers
# --------------------------------------------------------------------------- #

def _cpu_name() -> str:
    import platform

    return (
        os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.processor()
        or platform.machine()
        or "CPU"
    )


def _system_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:  # POSIX fallback
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return None


def _cpu_supports_bf16(torch: object) -> bool:
    try:
        return bool(torch.backends.cpu._is_amx_tile_supported())  # type: ignore[attr-defined]
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def detect(refresh: bool = False) -> DetectionReport:
    """Probe every runtime once and cache the result."""
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE

    devices: list[Device] = []
    statuses: list[RuntimeStatus] = []
    for probe in (_probe_torch, _probe_openvino, _probe_onnxruntime):
        try:
            found, status = probe()
        except Exception as exc:  # pragma: no cover - defensive, probes self-guard
            logger.warning("Probe %s failed unexpectedly: %s", probe.__name__, exc)
            continue
        devices.extend(found)
        statuses.extend(status)

    _CACHE = DetectionReport(tuple(devices), tuple(statuses))
    return _CACHE


def detect_devices(refresh: bool = False) -> list[Device]:
    """All reachable devices, ordered by :data:`DEFAULT_KIND_PRIORITY`.

    Within a kind, larger dedicated memory wins; the CPU always sorts last.
    """
    devices = list(detect(refresh).devices)
    kind_rank = {kind: rank for rank, kind in enumerate(DEFAULT_KIND_PRIORITY)}
    runtime_rank = {runtime: rank for rank, runtime in enumerate(RUNTIME_PREFERENCE)}
    devices.sort(
        key=lambda d: (
            kind_rank.get(d.kind, len(kind_rank)),
            -(d.total_memory_bytes or 0) if d.kind is not DeviceKind.CPU else 0,
            runtime_rank.get(d.runtime, len(runtime_rank)),
            d.index,
        )
    )
    return devices


def runtime_statuses(refresh: bool = False) -> list[RuntimeStatus]:
    """Per-runtime installation report, for ``imagegen doctor``."""
    return list(detect(refresh).statuses)


def select_device(selector: str = "auto", *, refresh: bool = False) -> Device:
    """Resolve a user selector to a concrete device.

    ``auto`` picks the best accelerator available (GPU > NPU > CPU). Any other
    value is matched against :meth:`Device.matches`; the first match in priority
    order wins.
    """
    devices = detect_devices(refresh)
    if not devices:  # pragma: no cover - the CPU entry is unconditional
        raise DeviceNotFoundError(
            "Aucun peripherique de calcul detecte.",
            hint="Verifiez que PyTorch est installe : uv pip install torch",
        )

    selector = (selector or "auto").strip()
    if selector.lower() in {"auto", ""}:
        return devices[0]

    for device in devices:
        if device.matches(selector):
            return device

    known = ", ".join(sorted({d.short_id for d in devices}))
    raise DeviceNotFoundError(
        f"Peripherique '{selector}' introuvable.",
        hint=f"Disponibles : {known}. Utilisez 'imagegen devices' pour la liste detaillee.",
    )
