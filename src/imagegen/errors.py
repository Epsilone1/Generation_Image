"""Exception hierarchy for imagegen.

Every error the CLI can surface to a user derives from :class:`ImageGenError` and
carries a ``hint``: the concrete next action to take. The CLI prints the message,
then the hint, and never a traceback unless ``--debug`` is passed.
"""

from __future__ import annotations


class ImageGenError(Exception):
    """Base class. ``hint`` holds the remediation shown to the user."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class ConfigurationError(ImageGenError):
    """Invalid or contradictory generation parameters."""


class UnknownModelError(ImageGenError):
    """The requested model key is not in the registry."""


class UnknownBackendError(ImageGenError):
    """The requested backend name is not registered."""


class BackendUnavailableError(ImageGenError):
    """A backend exists but cannot run here (missing package, missing hardware).

    This is the error a scaffolded backend raises when its runtime is not
    installed. It is expected and recoverable: the CLI can fall back to another
    backend instead of aborting.
    """


class BackendNotImplementedError(BackendUnavailableError):
    """The backend is wired into the registry but its inference path is a stub.

    Distinct from :class:`BackendUnavailableError` so that ``imagegen devices``
    can tell "install this package" apart from "not written yet".
    """


class DeviceNotFoundError(ImageGenError):
    """No accelerator matches the requested device selector."""


class ModelLoadError(ImageGenError):
    """The weights could not be downloaded, read, or instantiated."""


class GatedModelError(ModelLoadError):
    """The repository requires accepting a license and an authenticated token."""


class OutOfMemoryError(ImageGenError):
    """Inference ran out of device memory."""


class UnsupportedCapabilityError(ImageGenError):
    """The request needs a feature the selected backend/model does not provide.

    Example: an image-to-image request routed to a backend compiled for a single
    static text-to-image graph.
    """


class ImageIOError(ImageGenError):
    """An input image could not be read, or an output could not be written."""


class GenerationCancelled(ImageGenError):
    """The user aborted a generation.

    Raised from inside the per-step callback: diffusers has no supported abort
    mechanism, and nothing in the denoising loop catches exceptions, so this is
    how a Ctrl-C reaches the caller. The backend must call
    ``pipe.maybe_free_model_hooks()`` afterwards, since interrupting mid-loop
    under CPU offload can leave accelerate hooks in an inconsistent state.
    """

    def __init__(self, message: str = "Generation annulee.", hint: str | None = None) -> None:
        super().__init__(message, hint)
