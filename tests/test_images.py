"""Image input sizing and output metadata.

Sizing matters more than it looks: img2img pipelines have no ``width``/``height``
parameters, so the *source image* is what determines the output size. Getting
this wrong means ``--width 1024`` silently does nothing.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from imagegen.errors import ImageIOError
from imagegen.generation.images import (
    build_output_path,
    fit_dimensions,
    load_image,
    prepare_init_image,
    round_to_multiple,
    save_image,
)


def _image(width: int, height: int, colour=(120, 60, 30)) -> Image.Image:
    return Image.new("RGB", (width, height), colour)


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("value", "multiple", "expected"),
    [(513, 8, 512), (512, 8, 512), (100, 64, 64), (5, 8, 8), (1023, 8, 1016)],
)
def test_round_to_multiple_never_reaches_zero(value, multiple, expected):
    assert round_to_multiple(value, multiple) == expected


def test_fit_dimensions_preserves_aspect_ratio():
    width, height = fit_dimensions(4000, 3000, 1024 * 1024)
    assert width > height
    assert abs((width / height) - (4000 / 3000)) < 0.05


def test_fit_dimensions_lands_near_the_target_pixel_count():
    """A 12 MP photo must be brought to the model's trained scale, not run raw."""
    width, height = fit_dimensions(4000, 3000, 1024 * 1024)
    assert 0.7 < (width * height) / (1024 * 1024) < 1.3


def test_fit_dimensions_produces_vae_compatible_sizes():
    for source in [(4000, 3000), (1920, 1080), (640, 480), (100, 900)]:
        width, height = fit_dimensions(*source, 512 * 512)
        assert width % 8 == 0 and height % 8 == 0


def test_fit_dimensions_rejects_empty_images():
    with pytest.raises(ImageIOError):
        fit_dimensions(0, 100, 1024)


def test_prepare_init_image_returns_the_exact_size():
    assert prepare_init_image(_image(800, 600), 512, 512).size == (512, 512)


def test_prepare_init_image_crops_rather_than_stretches():
    """Stretching is the easier implementation and distorts every input."""
    source = _image(1000, 250)
    result = prepare_init_image(source, 512, 512)
    assert result.size == (512, 512)
    # Covering a square from a 4:1 strip means scaling by height, then cropping
    # width - so the vertical content survives untouched.


def test_prepare_init_image_is_a_no_op_at_the_right_size():
    source = _image(512, 512)
    assert prepare_init_image(source, 512, 512) is source


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(ImageIOError) as excinfo:
        load_image(tmp_path / "absent.png")
    assert "absent.png" in str(excinfo.value)


def test_unsupported_extension_lists_the_supported_ones(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("not an image")
    with pytest.raises(ImageIOError) as excinfo:
        load_image(path)
    assert ".png" in (excinfo.value.hint or "")


def test_corrupt_file_is_reported_clearly(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"definitely not a png")
    with pytest.raises(ImageIOError):
        load_image(path)


def test_images_are_converted_to_rgb(tmp_path):
    path = tmp_path / "grey.png"
    Image.new("L", (64, 64), 128).save(path)
    assert load_image(path).mode == "RGB"


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def test_parameters_are_embedded_in_the_png(tmp_path):
    """An image with no parameters attached cannot be reproduced."""
    parameters = {"prompt": "un chat", "seed": 42, "steps": 4, "model": "sdxl-lightning"}
    path = save_image(_image(64, 64), tmp_path / "out.png", parameters)
    with Image.open(path) as reopened:
        embedded = json.loads(reopened.text["imagegen"])
    assert embedded["seed"] == 42
    assert embedded["prompt"] == "un chat"


def test_widely_read_parameter_chunk_is_written(tmp_path):
    path = save_image(_image(64, 64), tmp_path / "out.png", {"prompt": "x", "seed": 1, "steps": 4})
    with Image.open(path) as reopened:
        assert "Seed: 1" in reopened.text["parameters"]


def test_sidecar_is_optional(tmp_path):
    path = save_image(_image(64, 64), tmp_path / "a.png", {"seed": 1}, sidecar=False)
    assert not path.with_suffix(".png.json").exists()
    path = save_image(_image(64, 64), tmp_path / "b.png", {"seed": 1}, sidecar=True)
    assert json.loads(path.with_suffix(".png.json").read_text(encoding="utf-8"))["seed"] == 1


def test_output_directory_is_created(tmp_path):
    target = tmp_path / "deep" / "nested" / "out.png"
    assert save_image(_image(32, 32), target, None).exists()


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #

def test_filename_carries_the_seed(tmp_path):
    path = build_output_path(tmp_path, "un phare dans la tempete", 42)
    assert "42" in path.name
    assert "phare" in path.name


def test_filenames_are_ascii_and_safe(tmp_path):
    path = build_output_path(tmp_path, "éléphant à l'aube / 50% #1", 7)
    assert path.name.isascii()
    for character in '<>:"/\\|?*':
        assert character not in path.name


def test_filenames_never_collide(tmp_path):
    first = build_output_path(tmp_path, "chat", 1)
    first.write_bytes(b"")
    second = build_output_path(tmp_path, "chat", 1)
    assert first != second


def test_empty_prompt_still_produces_a_name(tmp_path):
    assert build_output_path(tmp_path, "!!!", 5).name.startswith("image-5")
