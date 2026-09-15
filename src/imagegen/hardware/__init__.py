"""Hardware discovery: what accelerators exist here, and how to address them.

This package knows nothing about diffusion. It answers two questions:

* :func:`detect_devices` - which compute devices are reachable right now, across
  every runtime installed (PyTorch CUDA/XPU/MPS, DirectML, OpenVINO, ONNX Runtime).
* :func:`select_device` - given a user selector such as ``"auto"``, ``"cuda"``,
  ``"npu"`` or ``"ov:NPU"``, which one to use.

Backends consume :class:`Device` records; they never probe hardware themselves.
"""

from .detect import clear_detection_cache, detect_devices, select_device
from .device import Device, DeviceKind, Runtime, Vendor

__all__ = [
    "Device",
    "DeviceKind",
    "Runtime",
    "Vendor",
    "detect_devices",
    "select_device",
    "clear_detection_cache",
]
