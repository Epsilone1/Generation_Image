"""Core value objects exchanged between the CLI, the generator and the backends.

These types deliberately import neither torch nor diffusers: a backend running
on an NPU through OpenVINO or ONNX Runtime must consume them without pulling
PyTorch into the process.

The central distinction is between two kinds of parameter:

* :class:`PipelineSpec` - the **compile key**. Anything that, if changed,
  invalidates a compiled pipeline: resolution, batch size, whether
  classifier-free guidance is active, precision, device placement. On PyTorch
  these are free to vary per call; on every NPU path they are frozen at compile
  time (``Text2ImagePipeline.reshape(num_images, height, width, guidance_scale)``
  then ``.compile(device)``). Keeping them in a separate, hashable object is what
  lets the generator cache loaded pipelines and report a recompile instead of
  silently taking minutes.
* :class:`GenerationRequest` - the **per-call** parameters: prompt, seed, steps,
  strength. Free to change without touching the loaded pipeline.

Guidance sits in the compile key and not in the request because CFG doubles the
denoiser batch (unconditional + conditional). That is the non-obvious one, and
getting it wrong means a resolution-stable NPU pipeline still recompiles every
time the user nudges ``--guidance``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

    from .hardware.device import Device


class Mode(str, Enum):
    """What the request asks for, derived from whether a source image is given."""

    TEXT_TO_IMAGE = "txt2img"
    IMAGE_TO_IMAGE = "img2img"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


class Component(str, Enum):
    """A sub-model of a diffusion pipeline.

    Pipelines are not monolithic on the accelerator paths: OpenVINO compiles
    text encoder, UNet and VAE to independently chosen devices, and Qualcomm
    publishes them as three separate context binaries. Naming the parts is what
    makes heterogeneous placement and partial availability expressible.
    """

    TOKENIZER = "tokenizer"
    TEXT_ENCODER = "text_encoder"
    TEXT_ENCODER_2 = "text_encoder_2"
    #: UNet or DiT transformer, depending on the architecture.
    DENOISER = "denoiser"
    #: Required for img2img. Absent from most pre-compiled NPU bundles, which is
    #: why img2img support is a property of the resolved model, not the backend.
    VAE_ENCODER = "vae_encoder"
    VAE_DECODER = "vae_decoder"
    SCHEDULER = "scheduler"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


class SchedulerKind(str, Enum):
    """Backend-neutral scheduler names.

    Deliberately *not* diffusers class names. OpenVINO GenAI's ``Scheduler.Type``
    is a closed set (AUTO, DDIM, EULER_ANCESTRAL_DISCRETE, EULER_DISCRETE,
    FLOW_MATCH_EULER_DISCRETE, LCM, LMS_DISCRETE, PNDM) with no DPM++ or UniPC,
    and hard-compiled NPU bundles may allow none at all. Exposing the diffusers
    zoo as the public API would make ``--scheduler`` meaningless everywhere else;
    each backend maps this enum to what it has and warns on a downgrade.
    """

    AUTO = "auto"
    DDIM = "ddim"
    EULER = "euler"
    EULER_ANCESTRAL = "euler-a"
    DPMPP_2M = "dpmpp-2m"
    DPMPP_2M_KARRAS = "dpmpp-2m-karras"
    LCM = "lcm"
    LMS = "lms"
    PNDM = "pndm"
    FLOW_MATCH_EULER = "flow-match-euler"
    UNIPC = "unipc"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


class Effort(str, Enum):
    """How much compute to spend on one image.

    A single dial that moves several settings together, because they are not
    independent: on a distilled model, more steps without the matching adapter
    makes the image *worse*, and a step count that suits Euler-trailing is wrong
    for DPM++ with guidance. Each model maps the level to a coherent set - which
    adapter, how many steps, which guidance, whether a refinement pass runs.

    Explicit flags always win over the level: ``--effort max --steps 6`` runs six
    steps. The level fills in what the user did not pin.
    """

    DRAFT = "draft"
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    MAX = "max"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value

    @property
    def rank(self) -> int:
        return list(Effort).index(self)


#: Used when neither the CLI nor the caller says otherwise.
DEFAULT_EFFORT = Effort.BALANCED


class Precision(str, Enum):
    """Requested numeric precision, resolved per-device by the backend.

    ``AUTO`` lets the backend pick: fp16 on CUDA, bf16 where it is faster and
    safer, fp32 on CPU (fp16 on CPU is emulated and slower than fp32).
    """

    AUTO = "auto"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    NF4 = "nf4"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


#: Called once per denoising step with ``(step, total_steps)``. Backends that
#: cannot report intermediate progress call it once with ``(total, total)``.
ProgressCallback = Callable[[int, int], None]

#: Polled between steps; returning ``True`` aborts the generation. In the
#: signature from the start because retrofitting cancellation means changing
#: every backend's ``generate``.
CancelCallback = Callable[[], bool]


# --------------------------------------------------------------------------- #
# Device placement
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class DevicePlan:
    """Where each component runs.

    A single device is the common case and ``DevicePlan(device)`` expresses it.
    Heterogeneous placement exists because it is the *only* working Intel-NPU
    image-generation path today: text encoder on CPU, UNet on NPU, VAE decoder
    on GPU. A scalar ``device`` field would have to be redesigned the first time
    such hardware shows up.
    """

    default: Device
    #: Tuple of pairs rather than a dict, to stay hashable and usable in a cache key.
    overrides: tuple[tuple[Component, Device], ...] = ()

    def device_for(self, component: Component) -> Device:
        for name, device in self.overrides:
            if name is component:
                return device
        return self.default

    @property
    def is_uniform(self) -> bool:
        return not self.overrides

    @property
    def devices(self) -> tuple[Device, ...]:
        seen = [self.default]
        for _, device in self.overrides:
            if device not in seen:
                seen.append(device)
        return tuple(seen)

    def describe(self) -> str:
        if self.is_uniform:
            return self.default.short_id
        parts = [f"{c.value}={d.short_id}" for c, d in self.overrides]
        return f"{self.default.short_id} (+ {', '.join(parts)})"


# --------------------------------------------------------------------------- #
# The compile key
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """Everything that, if changed, may force a reload or a recompile.

    Hashable on purpose: the generator caches loaded backends by this value, and
    compiled-artefact caches on disk are keyed by :meth:`cache_key`.
    """

    model_key: str
    mode: Mode
    width: int
    height: int
    batch_size: int = 1
    #: Kept here, not in the request: CFG > 1 doubles the denoiser batch, so it
    #: is part of the compiled graph's shape on every static-shape runtime.
    guidance_scale: float = 0.0
    precision: Precision = Precision.AUTO
    scheduler: SchedulerKind = SchedulerKind.AUTO
    #: In the compile key because an effort level can select a different
    #: adapter, and swapping a fused LoRA means reloading the weights.
    effort: Effort = DEFAULT_EFFORT
    device_plan: DevicePlan | None = None

    @property
    def guidance_enabled(self) -> bool:
        return self.guidance_scale > 1.0

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    def cache_key(self, *, backend: str, runtime_version: str = "", driver: str = "") -> str:
        """Stable identifier for a compiled artefact on disk.

        Includes the driver and runtime versions because a compiled blob that
        survives a driver update tends to crash rather than degrade.
        """
        import hashlib

        payload = "|".join(
            str(part)
            for part in (
                backend,
                runtime_version,
                driver,
                self.model_key,
                self.mode.value,
                self.width,
                self.height,
                self.batch_size,
                f"{self.guidance_scale:.3f}",
                self.precision.value,
                self.scheduler.value,
                self.effort.value,
                self.device_plan.describe() if self.device_plan else "-",
            )
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"{self.model_key}-{self.mode.value}-{self.width}x{self.height}-{digest}"

    def describe(self) -> str:
        where = self.device_plan.describe() if self.device_plan else "?"
        cfg = f"cfg={self.guidance_scale:g}" if self.guidance_enabled else "sans cfg"
        return (
            f"{self.model_key} {self.mode.value} {self.width}x{self.height} "
            f"x{self.batch_size} {cfg} {self.precision.value} effort={self.effort.value} "
            f"sur {where}"
        )


# --------------------------------------------------------------------------- #
# The per-call request
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class GenerationRequest:
    """A single generation order.

    Fields left at ``None`` mean "use the model preset's default"; they are
    filled in by :meth:`with_defaults` before reaching a backend, so a backend
    never has to guess.
    """

    prompt: str
    negative_prompt: str | None = None

    #: Optional source image: its presence is what switches the request to
    #: img2img. A path or an already-loaded PIL image - the generator normalises
    #: it, and backends only ever see a PIL image.
    init_image: Path | Image | None = None
    strength: float = 0.65

    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    scheduler: SchedulerKind | None = None

    seed: int | None = None
    num_images: int = 1

    #: Free-form, backend-specific knobs, so an NPU backend can accept e.g.
    #: ``{"ov_config": {...}}`` without changing this dataclass.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ConfigurationError(
                "Le prompt est vide.",
                hint='Donnez un texte descriptif, par exemple : imagegen generate "un chat roux"',
            )
        if not 0.0 < self.strength <= 1.0:
            raise ConfigurationError(
                f"strength={self.strength} hors bornes.",
                hint="strength doit etre dans ]0, 1]. 0.3 = reste proche de l'image source, "
                "0.9 = s'en eloigne fortement.",
            )
        if self.num_images < 1:
            raise ConfigurationError("num_images doit valoir au moins 1.")
        if self.steps is not None and self.steps < 1:
            raise ConfigurationError("steps doit valoir au moins 1.")
        if self.guidance_scale is not None and self.guidance_scale < 0:
            raise ConfigurationError("guidance_scale ne peut pas etre negatif.")
        for name, value in (("width", self.width), ("height", self.height)):
            if value is None:
                continue
            if value < 64:
                raise ConfigurationError(f"{name}={value} est trop petit (minimum 64).")
            if value % 8 != 0:
                raise ConfigurationError(
                    f"{name}={value} n'est pas un multiple de 8.",
                    hint="Les VAE de diffusion travaillent au 1/8e : utilisez un multiple de 8 "
                    "(idealement de 64).",
                )
        if self.seed is not None and not 0 <= self.seed < 2**63:
            raise ConfigurationError("seed doit etre un entier positif sur 63 bits.")

    @property
    def mode(self) -> Mode:
        return Mode.IMAGE_TO_IMAGE if self.init_image is not None else Mode.TEXT_TO_IMAGE

    def with_defaults(self, **defaults: Any) -> GenerationRequest:
        """Return a copy where every ``None`` field is filled from ``defaults``."""
        patch = {
            key: value
            for key, value in defaults.items()
            if value is not None and getattr(self, key, "<missing>") is None
        }
        return replace(self, **patch) if patch else self

    @property
    def effective_steps(self) -> int:
        """Denoising steps actually run.

        img2img starts partway through the schedule: diffusers runs
        ``int(steps * strength)`` steps. When that product floors to zero the
        input image is returned unchanged, with no error - the single most
        confusing img2img failure, so it is computed explicitly here.
        """
        steps = self.steps or 0
        if self.mode is Mode.IMAGE_TO_IMAGE:
            return int(steps * self.strength)
        return steps


@dataclass(slots=True)
class GeneratedImage:
    """One produced image plus everything needed to reproduce it."""

    image: Image
    seed: int
    index: int = 0
    #: Flat, JSON-serialisable record of the parameters actually used.
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    """The outcome of one :class:`GenerationRequest`."""

    images: list[GeneratedImage]
    mode: Mode
    model_key: str
    backend: str
    device: str
    duration_s: float
    #: Time spent loading or compiling, reported separately so a first-run
    #: NPU compile does not look like a slow generation.
    load_s: float = 0.0
    paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def seconds_per_image(self) -> float:
        return self.duration_s / max(len(self.images), 1)


class Stopwatch:
    """Tiny context manager used to time loads and generations."""

    __slots__ = ("_start", "elapsed")

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start
