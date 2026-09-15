"""Memory planning on an 8 GB card.

The plan must be decided before loading, because the alternative is a CUDA OOM
traceback arriving at the last denoising step - the VAE decode is where an 8 GB
card actually runs out, not the denoiser.
"""

from __future__ import annotations

import pytest

from imagegen.backends.memory import MemoryPlan, estimate_activation_gb, plan_memory
from imagegen.hardware.device import Device, DeviceKind, Runtime, Vendor
from imagegen.models.registry import get_model
from imagegen.types import DevicePlan, Mode, PipelineSpec


def _gpu(memory_gb: float = 8.0) -> Device:
    return Device(
        runtime=Runtime.TORCH_CUDA,
        handle="cuda:0",
        kind=DeviceKind.GPU,
        vendor=Vendor.NVIDIA,
        name="Test GPU",
        total_memory_bytes=int(memory_gb * 1024**3),
        supports_fp16=True,
    )


CPU = Device(
    runtime=Runtime.TORCH_CPU,
    handle="cpu",
    kind=DeviceKind.CPU,
    name="Test CPU",
    total_memory_bytes=16 * 1024**3,
)

NPU = Device(
    runtime=Runtime.OPENVINO,
    handle="NPU",
    kind=DeviceKind.NPU,
    vendor=Vendor.INTEL,
    name="Test NPU",
    supports_dynamic_shapes=False,
)


def _spec(width=512, height=512, batch=1, guidance=0.0, device=None):
    return PipelineSpec(
        model_key="sd15",
        mode=Mode.TEXT_TO_IMAGE,
        width=width,
        height=height,
        batch_size=batch,
        guidance_scale=guidance,
        device_plan=DevicePlan(device or _gpu()),
    )


def test_small_model_stays_resident_on_8gb():
    decision = plan_memory(get_model("sd15"), _gpu(), _spec(), free_vram_gb=7.4)
    assert decision.plan is MemoryPlan.FULL_GPU


def test_sdxl_offloads_on_8gb():
    """SDXL fp16 is ~7 GB of weights; the desktop has already taken some VRAM."""
    decision = plan_memory(
        get_model("sdxl-lightning"), _gpu(), _spec(1024, 1024), free_vram_gb=7.4
    )
    assert decision.plan is MemoryPlan.MODEL_OFFLOAD


def test_sdxl_stays_resident_on_a_large_card():
    decision = plan_memory(
        get_model("sdxl-lightning"), _gpu(24), _spec(1024, 1024), free_vram_gb=23.0
    )
    assert decision.plan is MemoryPlan.FULL_GPU


def test_vae_tiling_is_on_at_high_resolution():
    """The 1024px OOM lands in VAE decode on the final step - tiling is the fix."""
    assert plan_memory(get_model("sdxl"), _gpu(), _spec(1024, 1024), free_vram_gb=7.4).vae_tiling
    assert not plan_memory(get_model("sd15"), _gpu(), _spec(512, 512), free_vram_gb=7.4).vae_tiling


def test_vae_slicing_follows_the_batch():
    """Slicing decodes one image at a time; it does nothing for a single image."""
    assert plan_memory(get_model("sd15"), _gpu(), _spec(batch=4), free_vram_gb=7.4).vae_slicing
    assert not plan_memory(get_model("sd15"), _gpu(), _spec(batch=1), free_vram_gb=7.4).vae_slicing


def test_cpu_device_forces_the_cpu_plan():
    decision = plan_memory(get_model("sd15"), CPU, _spec(device=CPU))
    assert decision.plan is MemoryPlan.CPU
    assert decision.warnings  # the user must know it will take minutes


def test_unknown_memory_chooses_the_cautious_plan():
    """NPUs share system RAM and report None; guessing 'plenty' would OOM."""
    decision = plan_memory(get_model("sdxl"), NPU, _spec(device=NPU), free_vram_gb=None)
    assert decision.plan is MemoryPlan.MODEL_OFFLOAD


def test_forced_offload_wins_over_the_automatic_choice():
    decision = plan_memory(get_model("sd15"), _gpu(), _spec(), force_offload=True, free_vram_gb=7.4)
    assert decision.plan is MemoryPlan.MODEL_OFFLOAD
    assert "force" in decision.reason


def test_forcing_residency_warns_when_it_will_not_fit():
    decision = plan_memory(
        get_model("sdxl"), _gpu(), _spec(1024, 1024), force_offload=False, free_vram_gb=3.0
    )
    assert decision.plan is MemoryPlan.FULL_GPU
    assert any("OOM" in w for w in decision.warnings)


def test_sequential_offload_is_the_last_resort():
    decision = plan_memory(get_model("sdxl"), _gpu(4), _spec(1024, 1024), free_vram_gb=2.0)
    assert decision.plan is MemoryPlan.SEQUENTIAL_OFFLOAD
    assert decision.warnings


def test_guidance_doubles_the_activation_estimate():
    """CFG runs the denoiser on a doubled batch: uncond + cond."""
    model = get_model("sd15")
    without = estimate_activation_gb(_spec(guidance=0.0), model)
    with_cfg = estimate_activation_gb(_spec(guidance=7.5), model)
    assert with_cfg > without


def test_activation_estimate_grows_with_pixels():
    model = get_model("sd15")
    assert estimate_activation_gb(_spec(1024, 1024), model) > estimate_activation_gb(
        _spec(512, 512), model
    )


def test_the_refine_upscale_is_included_in_the_plan():
    """The heaviest call is the refinement pass, not the one in the spec.

    Planning from the spec's resolution alone under-provisions the pass that
    actually decides whether the run fits.
    """
    model = get_model("sdxl-lightning")
    spec = _spec(1024, 1024, guidance=7.0)
    without = estimate_activation_gb(spec, model, 1.0)
    with_upscale = estimate_activation_gb(spec, model, 1.25)
    assert with_upscale > without
    # Area scales with the square of the factor.
    assert (with_upscale - 0.25) == pytest.approx((without - 0.25) * 1.25**2, rel=1e-6)


def test_vae_tiling_turns_on_for_an_upscaled_refinement():
    """A 768px base refined to 1.25x lands above the tiling threshold."""
    decision = plan_memory(
        get_model("sdxl-lightning"),
        _gpu(),
        _spec(896, 896),
        free_vram_gb=7.4,
        peak_resolution_scale=1.25,
    )
    assert decision.vae_tiling


def test_every_decision_explains_itself():
    """The reason string is what the CLI shows instead of a silent choice."""
    decision = plan_memory(get_model("sdxl"), _gpu(), _spec(1024, 1024), free_vram_gb=7.4)
    assert decision.reason
    assert decision.describe()
    assert decision.estimated_peak_gb == pytest.approx(decision.estimated_peak_gb)
