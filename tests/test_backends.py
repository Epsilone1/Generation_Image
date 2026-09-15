"""The backend contract, exercised without loading any weights.

These are the tests that keep the NPU path viable. Each one pins a behaviour
that, if it regressed, would only show up as a wrong image on hardware nobody
here has: a static-shape backend silently accepting an uncompiled resolution, a
no-CFG build silently dropping the negative prompt, an artefact without a VAE
encoder silently doing something other than img2img.
"""

from __future__ import annotations

import pytest

from imagegen.backends import backend_names, backend_report, get_backend_class, select_backend_class
from imagegen.backends.base import Backend, BackendCapabilities
from imagegen.errors import ConfigurationError, UnknownBackendError, UnsupportedCapabilityError
from imagegen.hardware.device import Device, DeviceKind, Runtime, Vendor
from imagegen.models.registry import get_model
from imagegen.types import (
    Component,
    DevicePlan,
    GenerationRequest,
    Mode,
    PipelineSpec,
    SchedulerKind,
)

GPU = Device(
    runtime=Runtime.TORCH_CUDA,
    handle="cuda:0",
    kind=DeviceKind.GPU,
    vendor=Vendor.NVIDIA,
    name="Test GPU",
    total_memory_bytes=8 * 1024**3,
    supports_fp16=True,
)
NPU = Device(
    runtime=Runtime.ONNXRUNTIME,
    handle="QNNExecutionProvider",
    kind=DeviceKind.NPU,
    vendor=Vendor.QUALCOMM,
    name="Hexagon",
    supports_dynamic_shapes=False,
)


def _spec(device=GPU, **overrides):
    base = {
        "model_key": "sd15",
        "mode": Mode.TEXT_TO_IMAGE,
        "width": 512,
        "height": 512,
        "device_plan": DevicePlan(device),
    }
    base.update(overrides)
    return PipelineSpec(**base)


class _FakeBackend(Backend):
    """A backend whose capabilities are dictated by the test."""

    name = "fake"
    runtimes = (Runtime.TORCH_CUDA, Runtime.ONNXRUNTIME)

    def __init__(self, model, spec, options=None, caps=None):
        super().__init__(model, spec, options)
        self._caps = caps or BackendCapabilities()

    @classmethod
    def availability(cls):
        from imagegen.backends.base import BackendAvailability

        return BackendAvailability(True)

    def load(self, progress=None):
        self._loaded = True

    def generate(self, request, progress=None, cancel=None):
        return []

    @property
    def capabilities(self):
        return self._caps


def _backend(caps=None, model="sd15", spec=None):
    return _FakeBackend(get_model(model), spec or _spec(), None, caps)


# --------------------------------------------------------------------------- #
# Capability derivation
# --------------------------------------------------------------------------- #

def test_img2img_is_derived_from_the_vae_encoder():
    """An ONNX export without vae_encoder/ can never do img2img."""
    without = BackendCapabilities(
        components=frozenset(
            {Component.TEXT_ENCODER, Component.DENOISER, Component.VAE_DECODER}
        )
    )
    assert without.text_to_image
    assert not without.image_to_image

    with_encoder = BackendCapabilities(
        components=frozenset(
            {
                Component.TEXT_ENCODER,
                Component.DENOISER,
                Component.VAE_DECODER,
                Component.VAE_ENCODER,
            }
        )
    )
    assert with_encoder.image_to_image


def test_img2img_request_on_a_decoder_only_artifact_is_refused():
    caps = BackendCapabilities(
        components=frozenset({Component.DENOISER, Component.VAE_DECODER})
    )
    backend = _backend(caps)
    request = GenerationRequest(prompt="x", init_image="a.png", steps=10)
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        backend.validate(request)
    assert "encodeur VAE" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Silent-failure guards
# --------------------------------------------------------------------------- #

def test_negative_prompt_on_a_no_cfg_build_warns():
    """A batch-1 compiled pipeline ignores it and returns a different image."""
    backend = _backend(BackendCapabilities(supports_negative_prompt=False))
    warnings = backend.validate(GenerationRequest(prompt="x", negative_prompt="flou"))
    assert any("negatif" in w for w in warnings)


def test_guidance_on_a_no_cfg_build_warns():
    backend = _backend(BackendCapabilities(supports_guidance=False))
    warnings = backend.validate(GenerationRequest(prompt="x", guidance_scale=7.5))
    assert any("guidance" in w for w in warnings)


def test_unavailable_scheduler_warns_rather_than_failing():
    caps = BackendCapabilities(supported_schedulers=(SchedulerKind.AUTO, SchedulerKind.EULER))
    backend = _backend(caps)
    warnings = backend.validate(
        GenerationRequest(prompt="x", scheduler=SchedulerKind.DPMPP_2M_KARRAS)
    )
    assert any("Scheduler" in w for w in warnings)


def test_zero_effective_steps_is_refused_with_the_fix():
    """diffusers does not guard this: it returns the input image untouched."""
    backend = _backend()
    request = GenerationRequest(prompt="x", init_image="a.png", steps=1, strength=0.5)
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        backend.validate(request)
    assert "--steps 2" in (excinfo.value.hint or "")


def test_sufficient_steps_pass():
    backend = _backend()
    assert backend.validate(
        GenerationRequest(prompt="x", init_image="a.png", steps=2, strength=0.5)
    ) is not None


# --------------------------------------------------------------------------- #
# Static shapes
# --------------------------------------------------------------------------- #

def test_uncompiled_resolution_is_refused_on_a_static_backend():
    caps = BackendCapabilities(dynamic_resolution=False, fixed_resolutions=((512, 512),))
    backend = _backend(caps)
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        backend.validate(GenerationRequest(prompt="x", width=768, height=768))
    assert "512x512" in (excinfo.value.hint or "")


def test_compiled_resolution_is_accepted():
    caps = BackendCapabilities(dynamic_resolution=False, fixed_resolutions=((512, 512),))
    backend = _backend(caps)
    assert backend.validate(GenerationRequest(prompt="x", width=512, height=512)) == []


def test_resolution_multiple_is_enforced():
    backend = _backend(BackendCapabilities(resolution_multiple=64))
    with pytest.raises(UnsupportedCapabilityError):
        backend.validate(GenerationRequest(prompt="x", width=520, height=512))


# --------------------------------------------------------------------------- #
# Reuse vs recompile
# --------------------------------------------------------------------------- #

def test_dynamic_backend_accepts_any_resolution():
    backend = _backend(BackendCapabilities(recompiles_on=frozenset()))
    assert backend.accepts(_spec(width=1024, height=1024))


def test_static_backend_rejects_a_changed_resolution():
    caps = BackendCapabilities(recompiles_on=frozenset({"width", "height"}))
    backend = _backend(caps)
    assert backend.accepts(_spec(width=512, height=512))
    assert not backend.accepts(_spec(width=768, height=768))


def test_static_backend_rejects_a_changed_guidance():
    """CFG doubles the denoiser batch, so it is part of the compiled graph."""
    caps = BackendCapabilities(recompiles_on=frozenset({"guidance_scale"}))
    backend = _backend(caps, spec=_spec(guidance_scale=0.0))
    assert backend.accepts(_spec(guidance_scale=0.0))
    assert not backend.accepts(_spec(guidance_scale=7.5))


def test_a_different_model_always_requires_a_reload():
    backend = _backend()
    assert not backend.accepts(_spec(model_key="sdxl"))


def test_a_different_device_always_requires_a_reload():
    backend = _backend()
    assert not backend.accepts(_spec(device=NPU))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_registry_lists_every_backend():
    assert {"torch", "openvino", "onnx"} <= set(backend_names())


def test_importing_backends_does_not_import_torch():
    """Startup must not depend on any optional runtime being installed."""
    import subprocess
    import sys

    code = (
        "import sys, imagegen.backends;"
        "assert 'torch' not in sys.modules, 'torch importe au chargement du registre';"
        "assert 'openvino' not in sys.modules;"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_availability_never_raises():
    for cls, availability in backend_report():
        assert isinstance(availability.available, bool)
        if not availability.available:
            assert availability.reason, f"{cls.name} indisponible sans raison"


def test_unavailable_backends_say_how_to_install_them():
    for cls, availability in backend_report():
        if not availability.available:
            assert availability.install_hint, f"{cls.name} n'indique pas comment l'installer"


def test_unknown_backend_lists_the_real_ones():
    with pytest.raises(UnknownBackendError) as excinfo:
        get_backend_class("tensorrt")
    assert "torch" in (excinfo.value.hint or "")


def test_torch_backend_is_chosen_for_a_cuda_device():
    assert select_backend_class(GPU).name == "torch"


def test_explicit_backend_must_support_the_device():
    """The mismatch is reported before any 'install this package' advice.

    Installing optimum would not make ONNX Runtime drive a torch CUDA device, so
    the install hint would send the user down the wrong path.
    """
    with pytest.raises(ConfigurationError) as excinfo:
        select_backend_class(GPU, preferred="onnx")
    assert "--device" in (excinfo.value.hint or "")


def test_torch_backend_declares_the_right_runtimes():
    cls = get_backend_class("torch")
    assert cls.supports_device(GPU)
    assert not cls.supports_device(NPU)
