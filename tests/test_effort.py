"""The effort ladder.

The point of a single dial is that the rungs stay *coherent*: steps, adapter,
scheduler and guidance move together. These tests pin the properties that make
that true, because each violation is a quality regression that produces an image
rather than an error.
"""

from __future__ import annotations

import pytest

from imagegen.models.registry import (
    Img2ImgMode,
    fixed_efforts,
    get_model,
    list_models,
    scaled_efforts,
)
from imagegen.types import DEFAULT_EFFORT, Effort, Mode, PipelineSpec

ALL_MODELS = list_models(include_opt_in=True)


def test_every_model_resolves_every_level():
    for spec in ALL_MODELS:
        for level in Effort:
            profile = spec.effort(level)
            assert profile.steps >= 1, f"{spec.key}/{level.value} sans etapes"


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_effort_never_decreases_work(spec):
    """Climbing the ladder must never ask for less compute.

    Steps alone are not the measure: a rung can keep the step count and add a
    refinement pass, or swap to a heavier adapter. What must hold is that total
    work is non-decreasing.
    """

    def work(level: Effort) -> float:
        """Denoiser passes, weighted by the canvas each one runs on."""
        profile = spec.effort(level)
        effective_refine = int(profile.refine_steps * profile.refine_strength)
        return profile.steps + effective_refine * profile.refine_scale**2

    values = [work(level) for level in Effort]
    assert values == sorted(values), f"{spec.key} n'est pas monotone : {values}"


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_step_locked_models_never_change_their_step_count(spec):
    """A distilled checkpoint run at the wrong step count degrades silently.

    Worse with an ancestral sampler, where ``step()`` injects fresh noise every
    iteration: raising the count re-rolls the image instead of refining it.
    """
    if not spec.steps_locked or spec.effort_profiles:
        return
    counts = {spec.effort(level).steps for level in Effort}
    assert counts == {spec.default_steps}, f"{spec.key} varie ses etapes malgre steps_locked"


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_refinement_is_only_asked_of_models_that_can_do_it(spec):
    """The refine pass is image-to-image; it needs a VAE encoder."""
    for level in Effort:
        if spec.effort(level).refines:
            assert spec.img2img_mode is not Img2ImgMode.UNSUPPORTED, (
                f"{spec.key}/{level.value} demande un affinage sans chemin img2img"
            )


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_refine_scale_stays_below_the_duplication_threshold(spec):
    """Above roughly 1.5x, SDXL starts duplicating the subject."""
    for level in Effort:
        assert spec.effort(level).refine_scale <= 1.5


def test_effective_refine_steps_actually_run():
    """A refine pass runs int(steps x strength) steps - it must exceed zero."""
    for spec in ALL_MODELS:
        for level in Effort:
            profile = spec.effort(level)
            if not profile.refines:
                continue
            effective = int(profile.refine_steps * profile.refine_strength)
            assert effective >= 4, (
                f"{spec.key}/{level.value} : {effective} etape(s) d'affinage reelles, "
                "trop peu pour ajouter du detail"
            )


def test_the_default_model_ladder_crosses_the_lightning_variants():
    """The whole ladder rides one set of base weights.

    Lower rungs swap a 394 MB Lightning LoRA; the top two drop it and run SDXL
    with real guidance. No rung costs a second checkpoint.
    """
    spec = get_model("sdxl-lightning")
    names = {
        level: (spec.effort(level).lora.weight_name if spec.effort(level).lora else None)
        for level in Effort
    }
    assert "2step" in names[Effort.DRAFT]
    assert names[Effort.FAST] is None  # the preset's own 4-step adapter
    assert "8step" in names[Effort.BALANCED]
    assert spec.effort(Effort.HIGH).drop_lora
    assert spec.effort(Effort.MAX).drop_lora


def test_dropping_the_adapter_restores_guidance():
    """Guidance 0 is a Lightning requirement, not an SDXL one.

    A rung that drops the distilled adapter must also restore classifier-free
    guidance and a scheduler that suits it, or the output is a washed-out
    28-step render with no guidance at all.
    """
    spec = get_model("sdxl-lightning")
    for level in (Effort.HIGH, Effort.MAX):
        profile = spec.effort(level)
        assert profile.guidance is not None and profile.guidance > 1.0
        assert profile.scheduler is not None
        # Negative micro-conditioning is inert without guidance, so it belongs
        # exactly here and nowhere lower on the ladder.
        assert "negative_original_size" in profile.call_kwargs


def test_negative_microcond_is_absent_where_guidance_is_off():
    """Setting it under guidance 0 would be a pure no-op dressed as a feature."""
    for spec in ALL_MODELS:
        for level in Effort:
            profile = spec.effort(level)
            guidance = profile.guidance if profile.guidance is not None else spec.default_guidance
            if guidance <= 1.0:
                assert "negative_original_size" not in profile.call_kwargs, (
                    f"{spec.key}/{level.value} : micro-conditionnement negatif sans guidage"
                )


def test_scaled_ladder_respects_a_floor():
    profiles = scaled_efforts(4, minimum=6)
    assert profiles[Effort.DRAFT].steps == 6


def test_fixed_ladder_buys_quality_with_refinement_not_steps():
    profiles = fixed_efforts(4)
    assert {p.steps for p in profiles.values()} == {4}
    assert not profiles[Effort.BALANCED].refines
    assert profiles[Effort.HIGH].refines
    assert profiles[Effort.MAX].refine_strength >= profiles[Effort.HIGH].refine_strength


def test_effort_is_part_of_the_compile_key():
    """Changing level can change the adapter, which means reloading weights."""

    def spec(level: Effort) -> PipelineSpec:
        return PipelineSpec(
            model_key="sdxl-lightning",
            mode=Mode.TEXT_TO_IMAGE,
            width=1024,
            height=1024,
            effort=level,
        )

    assert spec(Effort.BALANCED).cache_key(backend="torch") != spec(Effort.MAX).cache_key(
        backend="torch"
    )
    assert spec(Effort.BALANCED) != spec(Effort.MAX)


def test_default_level_is_in_the_middle():
    assert DEFAULT_EFFORT is Effort.BALANCED
    assert 0 < DEFAULT_EFFORT.rank < len(Effort) - 1
