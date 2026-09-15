"""Hardware detection must never raise, and must be honest about what it found.

Detection runs on every command, including on machines where an optional
runtime is half-installed or its driver is broken. A probe that raises would
make installing an NPU package break generation on a working GPU.
"""

from __future__ import annotations

import pytest

from imagegen.errors import DeviceNotFoundError
from imagegen.hardware.detect import (
    clear_detection_cache,
    detect_devices,
    runtime_statuses,
    select_device,
)
from imagegen.hardware.device import DeviceKind, Runtime


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_detection_cache()
    yield
    clear_detection_cache()


def test_detection_always_finds_the_cpu():
    """Whatever else fails, the CPU entry is unconditional."""
    devices = detect_devices()
    assert any(d.kind is DeviceKind.CPU for d in devices)


def test_detection_does_not_raise():
    assert detect_devices() is not None
    assert runtime_statuses() is not None


def test_accelerators_sort_before_the_cpu():
    """--device auto takes the first entry, so ordering is the policy."""
    devices = detect_devices()
    kinds = [d.kind for d in devices]
    if DeviceKind.GPU in kinds:
        assert kinds.index(DeviceKind.GPU) < kinds.index(DeviceKind.CPU)


def test_auto_selects_an_accelerator_when_present():
    devices = detect_devices()
    chosen = select_device("auto")
    assert chosen is devices[0]
    if any(d.is_accelerator for d in devices):
        assert chosen.is_accelerator


def test_cpu_is_always_selectable():
    assert select_device("cpu").kind is DeviceKind.CPU


def test_unknown_selector_lists_what_exists():
    with pytest.raises(DeviceNotFoundError) as excinfo:
        select_device("quantum-accelerator")
    assert "cpu" in (excinfo.value.hint or "")


def test_ids_round_trip_through_the_selector():
    for device in detect_devices():
        assert select_device(device.short_id).id == device.id


def test_non_device_providers_are_excluded():
    """Azure routes to a cloud endpoint and is always 'available' - not a device."""
    handles = {d.handle for d in detect_devices() if d.runtime is Runtime.ONNXRUNTIME}
    assert "AzureExecutionProvider" not in handles
    assert "CPUExecutionProvider" not in handles


def test_static_shape_devices_are_flagged():
    """Every NPU path compiles per resolution; the flag is what the CLI reads."""
    for device in detect_devices():
        if device.kind is DeviceKind.NPU:
            assert not device.supports_dynamic_shapes, (
                f"{device.short_id} est un NPU mais se declare a formes libres"
            )


def test_runtime_status_distinguishes_missing_from_broken():
    """'Not installed' and 'installed but failing' need different answers."""
    for status in runtime_statuses():
        if not status.installed:
            assert status.error or status.install_hint, (
                f"{status.runtime.value} est absent sans explication ni piste"
            )


def test_detection_is_cached():
    first = detect_devices()
    second = detect_devices()
    assert first[0] is second[0]
