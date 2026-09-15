"""Request validation and the compile-key semantics.

The img2img step arithmetic is tested here rather than left to the backend
because diffusers does *not* guard it: ``int(steps * strength) == 0`` runs an
empty denoising loop and returns a washed-out decode of the noised input, with
no error at all.
"""

from __future__ import annotations

import pytest

from imagegen.errors import ConfigurationError
from imagegen.hardware.device import Device, DeviceKind, Runtime, Vendor
from imagegen.types import (
    Component,
    DevicePlan,
    GenerationRequest,
    Mode,
    PipelineSpec,
    Precision,
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
    runtime=Runtime.OPENVINO,
    handle="NPU",
    kind=DeviceKind.NPU,
    vendor=Vendor.INTEL,
    name="Test NPU",
    supports_dynamic_shapes=False,
)


# --------------------------------------------------------------------------- #
# GenerationRequest
# --------------------------------------------------------------------------- #

def test_mode_follows_the_source_image():
    assert GenerationRequest(prompt="x").mode is Mode.TEXT_TO_IMAGE
    assert GenerationRequest(prompt="x", init_image="a.png").mode is Mode.IMAGE_TO_IMAGE


def test_empty_prompt_is_rejected():
    with pytest.raises(ConfigurationError):
        GenerationRequest(prompt="   ")


@pytest.mark.parametrize("strength", [0.0, -0.1, 1.5])
def test_strength_bounds(strength):
    with pytest.raises(ConfigurationError):
        GenerationRequest(prompt="x", strength=strength)


def test_dimensions_must_be_multiples_of_eight():
    with pytest.raises(ConfigurationError) as excinfo:
        GenerationRequest(prompt="x", width=513, height=512)
    assert "multiple de 8" in str(excinfo.value)


def test_dimensions_must_not_be_tiny():
    with pytest.raises(ConfigurationError):
        GenerationRequest(prompt="x", width=32, height=32)


def test_effective_steps_truncates():
    """int() truncates: 25 x 0.7 is 17 steps, not 18."""
    request = GenerationRequest(prompt="x", init_image="a.png", steps=25, strength=0.7)
    assert request.effective_steps == 17


def test_effective_steps_can_reach_zero():
    """The case diffusers does not guard, and the reason the generator does."""
    request = GenerationRequest(prompt="x", init_image="a.png", steps=1, strength=0.5)
    assert request.effective_steps == 0


def test_text_to_image_effective_steps_are_nominal():
    assert GenerationRequest(prompt="x", steps=20).effective_steps == 20


def test_with_defaults_only_fills_missing_fields():
    request = GenerationRequest(prompt="x", steps=10)
    filled = request.with_defaults(steps=30, guidance_scale=7.5)
    assert filled.steps == 10  # already set, untouched
    assert filled.guidance_scale == 7.5


# --------------------------------------------------------------------------- #
# PipelineSpec: the compile key
# --------------------------------------------------------------------------- #

def _spec(**overrides):
    base = {
        "model_key": "sd15",
        "mode": Mode.TEXT_TO_IMAGE,
        "width": 512,
        "height": 512,
        "device_plan": DevicePlan(GPU),
    }
    base.update(overrides)
    return PipelineSpec(**base)


def test_spec_is_hashable():
    """The generator caches loaded backends by spec, so it must hash."""
    assert len({_spec(), _spec()}) == 1


def test_guidance_participates_in_the_cache_key():
    """CFG doubles the denoiser batch, so it changes the compiled graph."""
    assert _spec(guidance_scale=0.0).cache_key(backend="ov") != _spec(guidance_scale=7.5).cache_key(
        backend="ov"
    )


def test_resolution_participates_in_the_cache_key():
    assert _spec(width=512).cache_key(backend="ov") != _spec(width=768).cache_key(backend="ov")


def test_driver_version_participates_in_the_cache_key():
    """A compiled blob surviving a driver update crashes rather than degrades."""
    spec = _spec()
    assert spec.cache_key(backend="ov", driver="1.0") != spec.cache_key(backend="ov", driver="1.1")


def test_guidance_enabled_threshold():
    assert not _spec(guidance_scale=1.0).guidance_enabled
    assert _spec(guidance_scale=1.5).guidance_enabled


# --------------------------------------------------------------------------- #
# DevicePlan
# --------------------------------------------------------------------------- #

def test_uniform_plan_returns_one_device():
    plan = DevicePlan(GPU)
    assert plan.is_uniform
    assert plan.device_for(Component.DENOISER) is GPU


def test_heterogeneous_plan_routes_per_component():
    """The only working Intel-NPU path: text encoder on CPU, denoiser on NPU."""
    plan = DevicePlan(GPU, overrides=((Component.DENOISER, NPU),))
    assert not plan.is_uniform
    assert plan.device_for(Component.DENOISER) is NPU
    assert plan.device_for(Component.VAE_DECODER) is GPU
    assert set(plan.devices) == {GPU, NPU}
    assert "denoiser=" in plan.describe()


# --------------------------------------------------------------------------- #
# Device selectors
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("selector", ["cuda:0", "cuda", "gpu", "nvidia", "torch-cuda:cuda:0"])
def test_gpu_selectors(selector):
    assert GPU.matches(selector)


@pytest.mark.parametrize("selector", ["npu", "ov:npu", "openvino", "intel"])
def test_npu_selectors(selector):
    assert NPU.matches(selector)


def test_selectors_do_not_cross_match():
    assert not GPU.matches("npu")
    assert not NPU.matches("cuda")


def test_short_ids_are_readable():
    assert GPU.short_id == "cuda:0"
    assert NPU.short_id == "ov:NPU"


def test_unknown_memory_is_none_not_zero():
    """NPUs share system RAM; None means unknown and must not be read as empty."""
    assert NPU.total_memory_gb is None
    assert GPU.total_memory_gb == pytest.approx(8.0)


def test_precision_enum_round_trips():
    assert Precision("fp16") is Precision.FP16
    assert SchedulerKind("dpmpp-2m-karras") is SchedulerKind.DPMPP_2M_KARRAS
