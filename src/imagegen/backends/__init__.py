"""Backend registry.

Backends are registered as *names*, not as imported classes: importing this
module must not import torch, openvino or onnxruntime. A machine missing any one
optional runtime would otherwise fail to start the CLI at all - and since the
whole point is to run on whatever accelerator is present, that failure mode is
exactly the one to design out.

Every heavy import lives inside a backend's :meth:`Backend.availability`, which
returns a reason string rather than raising.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from ..errors import ConfigurationError, UnknownBackendError
from ..hardware.device import Device
from .base import (
    Backend,
    BackendAvailability,
    BackendCapabilities,
    LoadOptions,
)
from .memory import MemoryDecision, MemoryPlan, plan_memory

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "Backend",
    "BackendAvailability",
    "BackendCapabilities",
    "LoadOptions",
    "MemoryDecision",
    "MemoryPlan",
    "plan_memory",
    "backend_names",
    "get_backend_class",
    "list_backend_classes",
    "backend_report",
    "select_backend_class",
]

#: name -> (module, class). Order is the fallback order when several backends
#: could serve a device.
_REGISTRY: dict[str, tuple[str, str]] = {
    "torch": ("imagegen.backends.torch_diffusers", "TorchDiffusersBackend"),
    "openvino": ("imagegen.backends.openvino_backend", "OpenVINOBackend"),
    "onnx": ("imagegen.backends.onnx_backend", "OnnxRuntimeBackend"),
}


def backend_names() -> list[str]:
    return list(_REGISTRY)


def get_backend_class(name: str) -> type[Backend]:
    """Import and return a backend class by name."""
    key = name.strip().lower()
    entry = _REGISTRY.get(key)
    if entry is None:
        raise UnknownBackendError(
            f"Backend '{name}' inconnu.",
            hint=f"Backends disponibles : {', '.join(_REGISTRY)}.",
        )
    module_name, class_name = entry
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def list_backend_classes() -> list[type[Backend]]:
    """Every backend class, skipping any that fails to import.

    An import failure here is a bug in that backend module, not a missing
    optional dependency (those are reported by ``availability``), so it is
    logged and stepped over rather than propagated.
    """
    classes: list[type[Backend]] = []
    for name in _REGISTRY:
        try:
            classes.append(get_backend_class(name))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Backend '%s' non importable : %s", name, exc)
    return classes


def backend_report() -> list[tuple[type[Backend], BackendAvailability]]:
    """``(class, availability)`` for every backend, for ``imagegen backends``."""
    report: list[tuple[type[Backend], BackendAvailability]] = []
    for cls in list_backend_classes():
        try:
            availability = cls.availability()
        except Exception as exc:  # pragma: no cover - availability must not raise
            availability = BackendAvailability(False, f"sonde en erreur : {exc}")
        report.append((cls, availability))
    return report


def select_backend_class(device: Device, preferred: str | None = None) -> type[Backend]:
    """Pick the backend to drive ``device``.

    ``preferred`` comes from ``--backend`` and is honoured even when another
    backend has a higher priority, because overriding the automatic choice is
    the whole point of the flag. It still has to support the device.
    """
    if preferred:
        cls = get_backend_class(preferred)
        # Device compatibility is checked first: it is the refusal no amount of
        # installing can fix, so telling the user to install optimum when the
        # real problem is that ONNX Runtime does not drive a CUDA torch device
        # would send them down the wrong path.
        if not cls.supports_device(device):
            raise ConfigurationError(
                f"Le backend '{cls.name}' ne pilote pas le peripherique {device.short_id} "
                f"({device.runtime.value}).",
                hint="Choisissez un peripherique compatible avec --device, ou laissez 'auto' "
                "decider. 'imagegen devices' liste les selecteurs valides.",
            )
        cls.availability().raise_if_unavailable(cls.name)
        return cls

    candidates = [cls for cls in list_backend_classes() if cls.supports_device(device)]
    candidates.sort(key=lambda cls: cls.priority, reverse=True)
    reasons: list[str] = []
    for cls in candidates:
        availability = cls.availability()
        if availability.available:
            return cls
        reasons.append(f"{cls.name} : {availability.reason}")

    detail = "; ".join(reasons) if reasons else "aucun backend ne pilote ce peripherique"
    raise UnknownBackendError(
        f"Aucun backend utilisable pour {device.short_id}.",
        hint=detail,
    )
