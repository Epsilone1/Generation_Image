"""The refinement pass, exercised against a stub backend.

The pass is a second image-to-image run over the first result. Everything that
can go wrong with it is arithmetic or plumbing - the upscale factor, the
effective step count, whether the second pass is even possible - so none of it
needs a real model to test.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from PIL import Image

from imagegen.backends.base import Backend, BackendAvailability, BackendCapabilities
from imagegen.generation.generator import ImageGenerator
from imagegen.hardware.device import Device, DeviceKind, Runtime, Vendor
from imagegen.models.registry import get_model
from imagegen.types import Component, Effort, GeneratedImage, GenerationRequest, Mode

GPU = Device(
    runtime=Runtime.TORCH_CUDA,
    handle="cuda:0",
    kind=DeviceKind.GPU,
    vendor=Vendor.NVIDIA,
    name="Stub GPU",
    total_memory_bytes=8 * 1024**3,
    supports_fp16=True,
)


class _RecordingBackend(Backend):
    """Returns flat images and records every request it was given."""

    name = "stub"
    runtimes = (Runtime.TORCH_CUDA,)

    calls: list[GenerationRequest] = []

    def __init__(self, model, spec, options=None):
        super().__init__(model, spec, options)
        self._caps = BackendCapabilities()

    @classmethod
    def availability(cls):
        return BackendAvailability(True)

    def load(self, progress=None):
        self._loaded = True

    def generate(self, request, progress=None, cancel=None):
        type(self).calls.append(request)
        size = (request.width or 512, request.height or 512)
        if request.mode is Mode.IMAGE_TO_IMAGE and request.init_image is not None:
            size = request.init_image.size
        return [
            GeneratedImage(
                image=Image.new("RGB", size, (60, 90, 120)),
                seed=request.seed or 0,
                index=index,
                parameters={},
            )
            for index in range(request.num_images)
        ]

    @property
    def capabilities(self):
        return self._caps


@pytest.fixture(autouse=True)
def _clear_calls():
    _RecordingBackend.calls = []
    yield
    _RecordingBackend.calls = []


def _generator(model_key: str, effort: Effort) -> ImageGenerator:
    return ImageGenerator(
        get_model(model_key), GPU, _RecordingBackend, effort=effort, translate="never"
    )


def test_balanced_runs_a_single_pass():
    generator = _generator("sdxl-lightning", Effort.BALANCED)
    result = generator.generate(GenerationRequest(prompt="a fox", seed=1))
    assert len(_RecordingBackend.calls) == 1
    assert result.images[0].image.size == (1024, 1024)


def test_max_runs_a_second_image_to_image_pass():
    generator = _generator("sdxl-lightning", Effort.MAX)
    generator.generate(GenerationRequest(prompt="a fox", seed=1))
    calls = _RecordingBackend.calls
    assert len(calls) == 2
    assert calls[0].mode is Mode.TEXT_TO_IMAGE
    assert calls[1].mode is Mode.IMAGE_TO_IMAGE


def _generator_with_upscale(scale: float) -> ImageGenerator:
    """A generator whose top rung upscales, whatever the shipped default is."""
    spec = get_model("sdxl-lightning")
    profiles = dict(spec.effort_profiles)
    profiles[Effort.MAX] = replace(profiles[Effort.MAX], refine_scale=scale)
    spec = replace(spec, effort_profiles=profiles)
    return ImageGenerator(spec, GPU, _RecordingBackend, effort=Effort.MAX, translate="never")


def test_the_shipped_ladder_does_not_upscale():
    """Measured on the 8 GB reference card: a 1.25x pass ran 10x longer.

    The mechanism stays; the shipped value does not use it. See
    MAX_REFINE_SCALE for the measurement.
    """
    assert get_model("sdxl-lightning").effort(Effort.MAX).refine_scale == 1.0


def test_the_refine_pass_upscales_when_the_profile_asks_for_it():
    generator = _generator_with_upscale(1.25)
    result = generator.generate(GenerationRequest(prompt="a fox", seed=1))
    refine = _RecordingBackend.calls[1]
    # 1024 x 1.25 = 1280, already a multiple of 64.
    assert refine.init_image.size == (1280, 1280)
    assert result.images[0].image.size == (1280, 1280)


def test_an_unaffordable_upscale_falls_back_with_a_warning(monkeypatch):
    """On a card that cannot hold the larger canvas, refine at native size.

    Silently spilling to system RAM would turn a 40-second generation into a
    ten-minute one, which is what this avoids.
    """
    monkeypatch.setattr(
        _RecordingBackend,
        "can_afford_upscale",
        lambda self, scale: (False, "pas assez de VRAM"),
    )
    generator = _generator_with_upscale(1.25)
    result = generator.generate(GenerationRequest(prompt="a fox", seed=1))
    assert _RecordingBackend.calls[1].init_image.size == (1024, 1024)
    assert any("Affinage ramene" in w for w in result.warnings)
    assert result.images[0].parameters["refine_scale"] == 1.0


def test_the_refine_pass_runs_enough_real_steps():
    """int(steps x strength) is what executes; below ~8 it adds noise, not detail."""
    generator = _generator("sdxl-lightning", Effort.MAX)
    generator.generate(GenerationRequest(prompt="a fox", seed=1))
    refine = _RecordingBackend.calls[1]
    assert refine.effective_steps >= 10


def test_the_refine_pass_keeps_the_seed():
    """Reproducibility must survive the second pass."""
    generator = _generator("sdxl-lightning", Effort.MAX)
    result = generator.generate(GenerationRequest(prompt="a fox", seed=4242))
    assert _RecordingBackend.calls[1].seed == 4242
    assert result.images[0].seed == 4242


def test_refined_images_record_how_they_were_refined():
    generator = _generator("sdxl-lightning", Effort.MAX)
    result = generator.generate(GenerationRequest(prompt="a fox", seed=1))
    parameters = result.images[0].parameters
    assert parameters["refine_scale"] == pytest.approx(1.0)
    assert parameters["effort"] == "max"
    assert parameters["refine_steps"] > 0
    assert parameters["final_size"] == [1024, 1024]


def test_each_image_of_a_batch_is_refined():
    generator = _generator("sdxl-lightning", Effort.MAX)
    result = generator.generate(GenerationRequest(prompt="a fox", seed=1, num_images=3))
    assert len(result.images) == 3
    # One text-to-image call, then one refine call per image.
    assert len(_RecordingBackend.calls) == 4


def test_refinement_is_skipped_with_a_warning_when_img2img_is_impossible(monkeypatch):
    """Never silently: a skipped pass means the requested effort was not spent."""
    generator = _generator("sdxl-lightning", Effort.MAX)
    decoder_only = BackendCapabilities(
        components=frozenset({Component.DENOISER, Component.VAE_DECODER})
    )
    monkeypatch.setattr(
        _RecordingBackend, "capabilities", property(lambda self: decoder_only)
    )
    result = generator.generate(GenerationRequest(prompt="a fox", seed=1))
    assert len(_RecordingBackend.calls) == 1
    assert any("affinage" in w for w in result.warnings)


def test_explicit_steps_override_the_effort_level():
    generator = _generator("sdxl-lightning", Effort.MAX)
    generator.generate(GenerationRequest(prompt="a fox", seed=1, steps=6))
    assert _RecordingBackend.calls[0].steps == 6


def test_explicit_guidance_overrides_the_effort_level():
    generator = _generator("sdxl-lightning", Effort.HIGH)
    generator.generate(GenerationRequest(prompt="a fox", seed=1, guidance_scale=2.5))
    assert _RecordingBackend.calls[0].guidance_scale == pytest.approx(2.5)


def test_changing_effort_forces_a_reload():
    """A rung can select a different adapter, and the old one was fused in."""
    generator = _generator("sdxl-lightning", Effort.FAST)
    generator.generate(GenerationRequest(prompt="a fox", seed=1))
    first = generator.backend
    generator.effort = Effort.BALANCED
    generator.generate(GenerationRequest(prompt="a fox", seed=1))
    assert generator.backend is not first


def test_same_effort_reuses_the_backend():
    generator = _generator("sdxl-lightning", Effort.FAST)
    generator.generate(GenerationRequest(prompt="a fox", seed=1))
    first = generator.backend
    generator.generate(GenerationRequest(prompt="a cat", seed=2))
    assert generator.backend is first
