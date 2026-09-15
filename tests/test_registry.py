"""The model catalogue must stay internally consistent.

These tests encode the traps the catalogue exists to avoid: a preset whose
img2img class is missing, a distilled model left at a guidance value that burns
the image, a required scheduler that nothing applies. Each one is a bug that
would otherwise surface as a bad image rather than an error.
"""

from __future__ import annotations

import pytest

from imagegen.models.registry import (
    DEFAULT_MODEL_KEY,
    LIGHTWEIGHT_MODEL_KEY,
    Commercial,
    Img2ImgMode,
    Tier,
    get_model,
    list_models,
    model_keys,
    recommend_model,
)
from imagegen.types import SchedulerKind

ALL_MODELS = list_models(include_opt_in=True)


def test_catalogue_is_not_empty():
    assert ALL_MODELS


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_keys_are_cli_safe(spec):
    assert spec.key == spec.key.lower()
    assert " " not in spec.key


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_img2img_mode_matches_declared_class(spec):
    """A separate-class model must name its class; a unified one must reuse it."""
    if spec.img2img_mode is Img2ImgMode.SEPARATE_CLASS:
        assert spec.img2img_class, f"{spec.key} declare une classe img2img separee mais ne la nomme pas"
        assert spec.img2img_class != spec.txt2img_class
    elif spec.img2img_mode is Img2ImgMode.SAME_CLASS_IMAGE_ARG:
        assert spec.img2img_class == spec.txt2img_class
    else:
        assert not spec.supports_img2img


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_distilled_models_have_guidance_off(spec):
    """Distilled models had CFG trained out; a non-zero scale burns the image.

    The tell is a very low step count. 7.5 leaking into one of these produces
    oversaturated output with no error, which is why it is checked here.
    """
    if spec.default_steps <= 4 and spec.family in {"sdxl", "sd15", "flux", "zimage"}:
        assert spec.default_guidance <= 2.0, (
            f"{spec.key} tourne en {spec.default_steps} etapes avec guidance="
            f"{spec.default_guidance} : valeur trop haute pour un modele distille"
        )


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_dimensions_are_vae_compatible(spec):
    assert spec.default_width % 8 == 0
    assert spec.default_height % 8 == 0


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_sizes_are_declared(spec):
    assert spec.download_gb > 0, f"{spec.key} n'annonce pas sa taille de telechargement"
    assert spec.weights_vram_gb > 0


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_non_commercial_models_are_opt_in(spec):
    """A user must never be silently committed to a non-commercial licence."""
    if spec.license.commercial is Commercial.NO:
        assert spec.opt_in, f"{spec.key} est non commercial et devrait etre opt-in"
        assert spec.warnings, f"{spec.key} est non commercial sans avertissement"


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_gated_models_require_a_token(spec):
    if spec.license.gated:
        assert spec.requires_token
        assert spec.opt_in, f"{spec.key} est gate : il ne peut pas etre un defaut silencieux"


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_quantized_models_declare_how(spec):
    if spec.tier is Tier.QUANTIZED:
        assert spec.quantization is not None, (
            f"{spec.key} ne tient pas en fp16 mais n'indique pas de strategie de quantification"
        )


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.key)
def test_required_schedulers_are_mappable(spec):
    if spec.scheduler is not None:
        assert isinstance(spec.scheduler.kind, SchedulerKind)
        assert spec.scheduler.kind is not SchedulerKind.AUTO


def test_default_model_is_shippable():
    """The default must work with no token, no licence trap and no quantization."""
    spec = get_model(DEFAULT_MODEL_KEY)
    assert not spec.opt_in
    assert not spec.license.gated
    assert spec.license.commercial is Commercial.YES
    assert spec.tier in (Tier.NATIVE, Tier.OFFLOAD)
    assert spec.supports_img2img


def test_lightweight_model_is_small():
    spec = get_model(LIGHTWEIGHT_MODEL_KEY)
    assert spec.weights_vram_gb < 4.0
    assert spec.tier is Tier.NATIVE


def test_recommendation_falls_back_on_unknown_memory():
    """A device that shares system RAM reports None, not a number."""
    assert recommend_model(None).key == LIGHTWEIGHT_MODEL_KEY
    assert recommend_model(4.0).key == LIGHTWEIGHT_MODEL_KEY
    assert recommend_model(8.0).key == DEFAULT_MODEL_KEY


def test_unknown_key_suggests_alternatives():
    from imagegen.errors import UnknownModelError

    with pytest.raises(UnknownModelError) as excinfo:
        get_model("sdxl-lightening")  # plausible misspelling
    assert "sdxl-lightning" in (excinfo.value.hint or "")


def test_min_steps_for_strength():
    """img2img runs int(steps * strength) steps; below one, nothing happens."""
    spec = get_model("sd15")
    assert spec.min_steps_for_strength(0.5) == 2
    assert spec.min_steps_for_strength(1.0) == 1
    assert spec.min_steps_for_strength(0.25) == 4


def test_keys_are_unique():
    keys = model_keys()
    assert len(keys) == len(set(keys))
