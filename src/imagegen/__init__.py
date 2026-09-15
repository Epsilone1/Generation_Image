"""imagegen - generation d'images locale a partir d'un prompt et, optionnellement, d'une image.

Point d'entree usuel::

    from imagegen import ImageGenerator, GenerationRequest

    gen = ImageGenerator.create(model="sdxl-turbo", device="auto")
    result = gen.generate(GenerationRequest(prompt="un phare dans la tempete"))

Le meme code s'execute sur GPU ou sur NPU : c'est le couple (backend, device)
resolu au chargement qui change, pas l'appel.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "GenerationRequest",
    "GenerationResult",
    "GeneratedImage",
    "Mode",
    "Precision",
    "ImageGenerator",
    "ImageGenError",
    "get_model",
    "list_models",
    "detect_devices",
    "select_device",
]


def __getattr__(name: str) -> object:
    """Lazy re-exports.

    Importing ``imagegen`` must stay instant: the CLI's ``devices`` and
    ``models`` commands should not pay for a torch import they do not need.
    """
    if name in {"GenerationRequest", "GenerationResult", "GeneratedImage", "Mode", "Precision"}:
        from . import types

        return getattr(types, name)
    if name == "ImageGenerator":
        from .generation.generator import ImageGenerator

        return ImageGenerator
    if name == "ImageGenError":
        from .errors import ImageGenError

        return ImageGenError
    if name in {"get_model", "list_models"}:
        from . import models

        return getattr(models, name)
    if name in {"detect_devices", "select_device"}:
        from . import hardware

        return getattr(hardware, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
