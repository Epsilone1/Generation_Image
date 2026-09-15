"""End-to-end tests that actually run a model.

Marked ``slow`` because they download weights and run real inference; excluded
from the default run. Invoke with::

    pytest tests -m slow

They exist because the two most load-bearing promises in this project were
flagged as *unverified* by the research that shaped the design: that diffusers
0.40 works on this torch/CUDA/Blackwell combination at all, and that a seed
reproduces a specific image out of a batch. Both are claims about behaviour, and
only running them settles it.
"""

from __future__ import annotations

import pytest

from imagegen.backends import LoadOptions
from imagegen.generation import ImageGenerator
from imagegen.types import GenerationRequest

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

#: The smallest preset, so the suite costs seconds rather than minutes.
MODEL = "sd15"


@pytest.fixture(scope="module")
def generator():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("aucun GPU CUDA disponible")
    gen = ImageGenerator.create(model=MODEL, device="auto", options=LoadOptions())
    yield gen
    gen.unload()


def test_text_to_image_produces_an_image(generator):
    result = generator.generate(
        GenerationRequest(prompt="a red cube on a table", steps=4, width=256, height=256, seed=1)
    )
    assert len(result.images) == 1
    image = result.images[0].image
    assert image.size == (256, 256)
    assert image.mode == "RGB"
    # A black or uniform image means the VAE overflowed to NaN.
    assert len(image.getcolors(maxcolors=100000) or []) > 50


def test_the_same_seed_reproduces_the_same_image(generator):
    request = lambda: GenerationRequest(  # noqa: E731
        prompt="a blue sphere", steps=4, width=256, height=256, seed=1234
    )
    first = generator.generate(request()).images[0].image
    second = generator.generate(request()).images[0].image
    assert first.tobytes() == second.tobytes()


def test_different_seeds_produce_different_images(generator):
    def run(seed: int):
        return generator.generate(
            GenerationRequest(prompt="a blue sphere", steps=4, width=256, height=256, seed=seed)
        ).images[0].image

    assert run(1).tobytes() != run(2).tobytes()


def test_one_image_from_a_batch_reproduces_alone(generator):
    """Re-running image *i* of a batch from ``seeds[i]`` gives the same picture.

    Not the same *bytes*, though, and that distinction is measured rather than
    assumed. diffusers draws the noise per batch element with its own generator,
    so the starting latent is identical - but running the denoiser on a batch of
    three selects different GEMM kernels and reduction orders than a batch of
    one, and floating-point addition is not associative. Measured here:
    mean absolute difference 2.5/255 against 73.4/255 for a different seed, with
    only 26% of pixels bit-identical.

    So the promise the seed in the filename carries is "the same image", exact
    only when the batch size also matches. ``test_the_same_seed_reproduces_the_
    same_image`` covers the exact case.
    """
    import numpy as np

    batch = generator.generate(
        GenerationRequest(
            prompt="a green pyramid", steps=4, width=256, height=256, seed=99, num_images=3
        )
    )
    assert len(batch.images) == 3
    target = batch.images[2]

    alone = generator.generate(
        GenerationRequest(
            prompt="a green pyramid", steps=4, width=256, height=256, seed=target.seed
        )
    ).images[0]
    unrelated = generator.generate(
        GenerationRequest(
            prompt="a green pyramid", steps=4, width=256, height=256, seed=target.seed + 500
        )
    ).images[0]

    reference = np.asarray(target.image, dtype=np.int16)
    reproduced = np.asarray(alone.image, dtype=np.int16)
    different = np.asarray(unrelated.image, dtype=np.int16)

    same_seed_delta = np.abs(reference - reproduced).mean()
    other_seed_delta = np.abs(reference - different).mean()

    assert same_seed_delta < 10, f"la graine ne reproduit pas l'image (ecart {same_seed_delta:.1f})"
    assert other_seed_delta > 5 * same_seed_delta, (
        "une autre graine devrait donner une image franchement differente"
    )


def test_image_to_image_preserves_the_source_size(generator, tmp_path):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (384, 256), (200, 120, 60)).save(source)

    result = generator.generate(
        GenerationRequest(
            prompt="a stormy sea",
            init_image=source,
            strength=0.6,
            steps=8,
            width=256,
            height=256,
            seed=5,
        )
    )
    assert result.images[0].image.size == (256, 256)
    assert result.mode.value == "img2img"


def test_image_to_image_actually_changes_the_image(generator, tmp_path):
    """A zero-step img2img returns the input untouched, with no error."""
    from PIL import Image

    source_image = Image.new("RGB", (256, 256), (200, 120, 60))
    source = tmp_path / "flat.png"
    source_image.save(source)

    result = generator.generate(
        GenerationRequest(
            prompt="a dense jungle", init_image=source, strength=0.8, steps=8, seed=3
        )
    )
    assert result.images[0].image.tobytes() != source_image.tobytes()


def test_effective_steps_are_recorded(generator, tmp_path):
    from PIL import Image

    source = tmp_path / "s.png"
    Image.new("RGB", (256, 256), (10, 10, 10)).save(source)

    result = generator.generate(
        GenerationRequest(
            prompt="a city", init_image=source, strength=0.5, steps=10, seed=1
        )
    )
    parameters = result.images[0].parameters
    assert parameters["effective_steps"] == 5  # int(10 * 0.5)
    assert parameters["steps"] == 10


def test_progress_reports_the_real_step_count(generator):
    """For img2img the total is int(steps*strength): a bar on the nominal
    count stops at 60% and looks broken."""
    seen: list[tuple[int, int]] = []
    generator.generate(
        GenerationRequest(prompt="a lantern", steps=4, width=256, height=256, seed=2),
        progress=lambda step, total: seen.append((step, total)),
    )
    assert seen
    assert seen[-1][0] == seen[-1][1] == 4


def test_cancelling_raises_rather_than_returning_a_partial_image(generator):
    from imagegen.errors import GenerationCancelled

    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    with pytest.raises(GenerationCancelled):
        generator.generate(
            GenerationRequest(prompt="a tower", steps=20, width=256, height=256, seed=4),
            cancel=cancel,
        )


def test_an_overlong_prompt_is_reported_as_truncated(generator):
    """CLIP reads 77 tokens and drops the rest, silently.

    The style tags people append are exactly what gets lost, so the loss has to
    be named rather than left to be discovered in the image.
    """
    # 12 repetitions is only 70 tokens - still under the limit. 30 is 160.
    long_prompt = (
        "a red fox in a snowy forest, " + "extremely detailed intricate ornate, " * 30
    )
    result = generator.generate(
        GenerationRequest(prompt=long_prompt, steps=2, width=256, height=256, seed=1)
    )
    assert any("tronque" in w and "77" in w for w in result.warnings)


def test_a_normal_prompt_is_not_reported_as_truncated(generator):
    result = generator.generate(
        GenerationRequest(prompt="a red fox", steps=2, width=256, height=256, seed=1)
    )
    assert not any("tronque" in w for w in result.warnings)


def test_a_negative_prompt_without_guidance_is_reported(generator):
    """diffusers only runs the unconditional branch when guidance > 1.

    Below that the negative prompt is discarded and an image comes back anyway -
    just not the one that was asked for.
    """
    result = generator.generate(
        GenerationRequest(
            prompt="a red fox",
            negative_prompt="blurry, low quality",
            guidance_scale=0.0,
            steps=2,
            width=256,
            height=256,
            seed=1,
        )
    )
    assert any("negatif" in w for w in result.warnings)


def test_a_negative_prompt_with_guidance_is_not_flagged(generator):
    result = generator.generate(
        GenerationRequest(
            prompt="a red fox",
            negative_prompt="blurry",
            guidance_scale=7.0,
            steps=2,
            width=256,
            height=256,
            seed=1,
        )
    )
    assert not any("negatif" in w for w in result.warnings)


def test_unload_actually_frees_vram():
    """Dropping references is not enough - the offload hooks must go too.

    ``enable_model_cpu_offload`` installs accelerate hooks holding every
    sub-model. Without removing them, a second load in the same process runs
    against a nearly full pool: measured 7.9 GB of 8 GB still occupied, and the
    next generation took 38 s instead of 7 s because the Windows driver spilled
    to system RAM rather than raising. Switching model or effort level takes
    exactly this path, so it is worth a test of its own.
    """
    import gc

    import torch

    from imagegen.backends import LoadOptions
    from imagegen.generation import ImageGenerator

    gc.collect()
    torch.cuda.empty_cache()
    before = torch.cuda.mem_get_info()[0]

    gen = ImageGenerator.create(model=MODEL, device="auto", options=LoadOptions())
    gen.generate(GenerationRequest(prompt="a cube", steps=2, width=256, height=256, seed=1))
    during = torch.cuda.mem_get_info()[0]
    assert during < before, "le modele ne semble pas avoir ete charge en VRAM"

    gen.unload()
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.mem_get_info()[0]

    leaked_gb = (before - after) / (1024**3)
    assert leaked_gb < 0.5, f"{leaked_gb:.2f} Go non liberes apres unload()"


def test_backend_is_reused_across_requests(generator):
    """A resolution change must not reload on a dynamic-shape backend."""
    generator.generate(
        GenerationRequest(prompt="a", steps=2, width=256, height=256, seed=1)
    )
    backend = generator.backend
    generator.generate(
        GenerationRequest(prompt="b", steps=2, width=320, height=256, seed=1)
    )
    assert generator.backend is backend
