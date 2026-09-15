"""Image input and output.

Two jobs, both of which are where a generator quietly goes wrong:

* **Sizing the source image.** img2img pipelines have no ``width``/``height``
  parameters - the output size is the *input* size, rounded down to a multiple of
  the VAE scale factor. So "generate at 1024" for an image-to-image request is a
  statement about the source image, and resizing it is the only way to honour it.

* **Recording what produced the file.** A generated image with no parameters
  attached cannot be reproduced. Every PNG carries its seed and settings in text
  chunks, with an optional JSON sidecar for tools that do not read them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ImageIOError

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

#: Extensions PIL reads reliably for our purposes.
_READABLE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


def load_image(source: Path | str) -> Image:
    """Read an image from disk as RGB, with errors that name the file."""
    from PIL import Image as PILImage
    from PIL import UnidentifiedImageError

    path = Path(source)
    if not path.exists():
        raise ImageIOError(
            f"Image source introuvable : {path}",
            hint="Verifiez le chemin. Les chemins relatifs partent du repertoire courant.",
        )
    if path.suffix.lower() not in _READABLE:
        raise ImageIOError(
            f"Format non pris en charge : {path.suffix or '(aucune extension)'}",
            hint=f"Formats acceptes : {', '.join(sorted(_READABLE))}.",
        )
    try:
        with PILImage.open(path) as handle:
            handle.load()
            return handle.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ImageIOError(
            f"Fichier illisible en tant qu'image : {path}",
            hint="Le fichier est peut-etre corrompu ou n'est pas une image.",
        ) from exc
    except OSError as exc:
        raise ImageIOError(f"Lecture de '{path}' impossible : {exc}") from exc


def round_to_multiple(value: int, multiple: int) -> int:
    """Round down to a multiple, never below one multiple."""
    if multiple <= 1:
        return max(value, 1)
    return max(multiple, (value // multiple) * multiple)


def fit_dimensions(
    width: int,
    height: int,
    target_pixels: int,
    multiple: int = 8,
) -> tuple[int, int]:
    """Scale a size to roughly ``target_pixels`` while keeping its aspect ratio.

    Used when the user gives a source image but no explicit output size: an
    arbitrary photo should be brought to the model's trained scale, not run at
    whatever the camera produced. A 4000x3000 photo fed to SDXL at native size
    would OOM; at 512 it would look nothing like SDXL's training distribution.
    """
    if width <= 0 or height <= 0:
        raise ImageIOError(f"Dimensions d'image invalides : {width}x{height}")
    scale = (target_pixels / (width * height)) ** 0.5
    return (
        round_to_multiple(int(width * scale), multiple),
        round_to_multiple(int(height * scale), multiple),
    )


def prepare_init_image(
    image: Image,
    width: int,
    height: int,
) -> Image:
    """Resize a source image to exactly ``width`` x ``height``.

    Aspect ratio is preserved by scaling to cover the target and centre-cropping
    the excess. Stretching would be the easier implementation and would visibly
    distort every non-matching input.
    """
    from PIL import Image as PILImage

    if image.size == (width, height):
        return image

    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    scaled = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
    resized = image.resize(scaled, PILImage.LANCZOS)

    left = (scaled[0] - width) // 2
    top = (scaled[1] - height) // 2
    return resized.crop((left, top, left + width, top + height))


def save_image(
    image: Image,
    path: Path,
    parameters: dict[str, Any] | None = None,
    *,
    sidecar: bool = False,
) -> Path:
    """Write an image, embedding its generation parameters.

    PNG metadata travels with the file; the sidecar is for tools that ignore
    text chunks. Neither is written for lossy formats that would drop it.
    """
    from PIL import PngImagePlugin

    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ImageIOError(f"Creation du dossier '{path.parent}' impossible : {exc}") from exc

    try:
        if path.suffix.lower() == ".png" and parameters:
            info = PngImagePlugin.PngInfo()
            info.add_text("imagegen", json.dumps(parameters, ensure_ascii=False, default=str))
            # Also written in the convention other tools read.
            info.add_text("parameters", _as_a1111_parameters(parameters))
            image.save(path, pnginfo=info)
        else:
            image.save(path)
    except OSError as exc:
        raise ImageIOError(f"Ecriture de '{path}' impossible : {exc}") from exc

    if sidecar and parameters:
        meta_path = path.with_suffix(path.suffix + ".json")
        try:
            meta_path.write_text(
                json.dumps(parameters, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - non-fatal
            raise ImageIOError(f"Ecriture des metadonnees '{meta_path}' impossible : {exc}") from exc
    return path


def _as_a1111_parameters(parameters: dict[str, Any]) -> str:
    """Render parameters in the widely-read text-chunk convention."""
    prompt = parameters.get("prompt", "")
    negative = parameters.get("negative_prompt")
    fields = [
        f"Steps: {parameters.get('steps')}",
        f"Sampler: {parameters.get('scheduler')}",
        f"CFG scale: {parameters.get('guidance_scale')}",
        f"Seed: {parameters.get('seed')}",
        f"Size: {parameters.get('width')}x{parameters.get('height')}",
        f"Model: {parameters.get('model')}",
    ]
    lines = [str(prompt)]
    if negative:
        lines.append(f"Negative prompt: {negative}")
    lines.append(", ".join(fields))
    return "\n".join(lines)


def build_output_path(
    directory: Path,
    prompt: str,
    seed: int,
    index: int = 0,
    extension: str = ".png",
) -> Path:
    """A unique, descriptive filename that never collides.

    The seed is in the name because it is what a user needs to reproduce the
    image, and looking it up in metadata is a step too many.
    """
    slug = _slugify(prompt) or "image"
    base = f"{slug}-{seed}"
    if index:
        base = f"{base}-{index}"
    candidate = directory / f"{base}{extension}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{base}-{counter}{extension}"
        counter += 1
    return candidate


def _slugify(text: str, max_length: int = 48) -> str:
    import re
    import unicodedata

    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit("-", 1)[0]
    return slug
