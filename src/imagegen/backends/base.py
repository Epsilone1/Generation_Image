"""The backend contract.

A *backend* pairs an inference runtime with a model format: PyTorch + diffusers,
OpenVINO IR, ONNX Runtime graphs. Adding NPU support later means adding a
backend, never touching the generator or the CLI.

The contract is shaped by what NPUs impose, not by what PyTorch makes easy.
Five constraints drove it:

1. **Shapes and guidance are a compile key, not parameters.** Every NPU path
   compiles the denoiser for one fixed resolution *and* one batch size - and CFG
   doubles that batch, so ``guidance_scale`` is an argument to OpenVINO's
   ``reshape()``. Hence :class:`~imagegen.types.PipelineSpec`, the separate
   :meth:`Backend.accepts` check, and :attr:`BackendCapabilities.recompiles_on`.

2. **Capabilities depend on the resolved model, not the backend class.**
   img2img needs a VAE *encoder*; ``optimum-cli export onnx`` emits one only for
   the img2img task, and Qualcomm's published bundles ship text encoder + UNet +
   VAE decoder only. So the same backend does img2img for one model and not
   another. :attr:`BackendCapabilities.components` is the source of truth.

3. **Classifier-free guidance is not always available.** A pipeline compiled at
   batch=1 silently ignores the negative prompt - an image comes back, just not
   the requested one. Capabilities are computed per compile, and the generator
   warns rather than letting it pass.

4. **Per-submodel placement.** The only working Intel-NPU path is heterogeneous
   (text encoder on CPU, UNet on NPU, VAE on GPU), so the backend receives a
   :class:`~imagegen.types.DevicePlan`, not a device.

5. **Runtimes return different things.** OpenVINO GenAI hands back an
   ``ov.Tensor``, ORT numpy, torch PIL. :meth:`Backend.generate` always returns
   PIL images, and always accepts PIL for the source image.

Backend classes are registered lazily: importing ``imagegen.backends`` must not
import torch, openvino or onnxruntime. Every probe lives inside
:meth:`Backend.availability` and returns a reason string instead of raising.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import BackendUnavailableError, UnsupportedCapabilityError
from ..hardware.device import Device, Runtime
from ..types import (
    CancelCallback,
    Component,
    GeneratedImage,
    GenerationRequest,
    Mode,
    PipelineSpec,
    Precision,
    ProgressCallback,
    SchedulerKind,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..models.registry import ModelSpec


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """What a backend can do *for one resolved model and compile config*.

    Defaults describe a fully capable PyTorch backend; constrained backends
    override the relevant flags and the generator adapts or refuses clearly.
    """

    #: Sub-models actually present in the resolved artefact. img2img is derived
    #: from this, never assumed.
    components: frozenset[Component] = frozenset(
        {
            Component.TOKENIZER,
            Component.TEXT_ENCODER,
            Component.DENOISER,
            Component.VAE_ENCODER,
            Component.VAE_DECODER,
            Component.SCHEDULER,
        }
    )

    supports_negative_prompt: bool = True
    supports_guidance: bool = True
    supports_seed: bool = True
    #: Bit-exact reproduction *within this backend, at this exact config*.
    #: Two caveats, both measured rather than assumed:
    #: across backends a seed never matches (quantization and different RNGs),
    #: and even within the torch backend a change of batch size gives the same
    #: image but not the same bytes - batched GEMMs pick different kernels and
    #: floating-point addition is not associative (measured: mean delta 2.5/255
    #: for the same seed at a different batch size, against 73/255 for a
    #: different seed).
    deterministic_seed: bool = True
    supports_batch: bool = True
    supports_progress_callback: bool = True
    supports_cancel: bool = True

    #: ``False`` means the graph is compiled per resolution.
    dynamic_resolution: bool = True
    #: Allow-list when ``dynamic_resolution`` is ``False``. AMD's "dynamic
    #: resolution" bundles cover a *set* of shapes, not a single one, so this is
    #: a tuple rather than one pair.
    fixed_resolutions: tuple[tuple[int, int], ...] = ()
    #: Imposed by the VAE downsampling factor (8 for SD/SDXL) and by tiling.
    resolution_multiple: int = 8

    #: Spec fields whose change forces a reload/recompile. Empty on PyTorch.
    recompiles_on: frozenset[str] = frozenset()
    #: Whether a compiled artefact can be cached on disk between runs.
    supports_compiled_cache: bool = False

    supported_schedulers: tuple[SchedulerKind, ...] = (SchedulerKind.AUTO,)
    supported_precisions: tuple[Precision, ...] = (Precision.FP32,)

    # --- Extension points: declared now, implemented later ----------------- #
    supports_lora: bool = False
    supports_controlnet: bool = False
    supports_ip_adapter: bool = False
    supports_inpainting: bool = False

    @property
    def text_to_image(self) -> bool:
        return Component.DENOISER in self.components and Component.VAE_DECODER in self.components

    @property
    def image_to_image(self) -> bool:
        """Needs a VAE encoder to turn the source image into latents."""
        return self.text_to_image and Component.VAE_ENCODER in self.components

    def describe(self) -> list[str]:
        """Human-readable capability lines for ``imagegen info``."""
        lines = [
            f"txt2img          : {'oui' if self.text_to_image else 'non'}",
            f"img2img          : {'oui' if self.image_to_image else 'non'}"
            + ("" if self.image_to_image else "  (aucun encodeur VAE dans l'artefact)"),
            f"prompt negatif   : {'oui' if self.supports_negative_prompt else 'non'}",
            f"guidage (CFG)    : {'oui' if self.supports_guidance else 'non'}",
            f"graine reprod.   : {'oui' if self.deterministic_seed else 'non'}",
        ]
        if self.dynamic_resolution:
            lines.append(f"resolution       : libre (multiple de {self.resolution_multiple})")
        else:
            shapes = ", ".join(f"{w}x{h}" for w, h in self.fixed_resolutions) or "inconnue"
            lines.append(f"resolution       : fixe ({shapes})")
        if self.recompiles_on:
            lines.append("recompile si     : " + ", ".join(sorted(self.recompiles_on)))
        lines.append("schedulers       : " + ", ".join(s.value for s in self.supported_schedulers))
        lines.append("precisions       : " + ", ".join(p.value for p in self.supported_precisions))
        extras = [
            name
            for name, ok in (
                ("LoRA", self.supports_lora),
                ("ControlNet", self.supports_controlnet),
                ("IP-Adapter", self.supports_ip_adapter),
                ("inpainting", self.supports_inpainting),
            )
            if ok
        ]
        lines.append("extensions       : " + (", ".join(extras) if extras else "aucune"))
        return lines


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    """Whether a backend can be used on this machine, and if not, why.

    Three distinct states matter and are routinely conflated:
    the package is missing, the package is installed but built without the
    execution provider, or the hardware simply is not there.
    """

    available: bool
    reason: str = ""
    install_hint: str | None = None
    #: ``True`` when the runtime is present but the inference path is a stub.
    implemented: bool = True

    def raise_if_unavailable(self, backend_name: str) -> None:
        if self.available:
            return
        raise BackendUnavailableError(
            f"Backend '{backend_name}' indisponible : {self.reason}",
            hint=self.install_hint,
        )


@dataclass(slots=True)
class LoadOptions:
    """How weights are loaded, as opposed to what is generated.

    Torch-only memory knobs live here rather than in the typed interface,
    because they are meaningless on three of the four backends.
    """

    #: ``None`` lets the backend decide from free VRAM; ``True``/``False`` force it.
    cpu_offload: bool | None = None
    sequential_offload: bool = False
    vae_slicing: bool | None = None
    vae_tiling: bool | None = None
    attention_slicing: bool = False
    compile_model: bool = False

    #: Largest resolution the run will actually touch, relative to the spec's.
    #: A refinement pass upscales before its second denoising pass, so planning
    #: memory from the spec alone would under-provision the heaviest call.
    peak_resolution_scale: float = 1.0

    cache_dir: str | None = None
    local_files_only: bool = False
    token: str | None = None
    safety_checker: bool = False
    #: Where compiled artefacts (OpenVINO blobs, ORT context binaries) are cached.
    compiled_cache_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """One runtime + model-format pairing.

    Lifecycle::

        backend = SomeBackend(model_spec, pipeline_spec, options)
        backend.load()                       # weights -> device (+ compile)
        images = backend.generate(request)   # per-call parameters only
        backend.unload()

    A change of resolution, batch size or guidance produces a *new*
    :class:`PipelineSpec`; the generator asks :meth:`accepts` before reusing a
    loaded backend and reloads when the answer is no.
    """

    #: Registry key, also the value accepted by ``--backend``.
    name: ClassVar[str] = ""
    #: Device runtimes this backend can drive.
    runtimes: ClassVar[tuple[Runtime, ...]] = ()
    #: Higher wins when several backends can serve the same device.
    priority: ClassVar[int] = 0
    #: Artefact format consumed ("diffusers", "openvino", "onnx").
    model_format: ClassVar[str] = "diffusers"
    #: Shown by ``imagegen backends``.
    description: ClassVar[str] = ""

    def __init__(
        self,
        model: ModelSpec,
        pipeline_spec: PipelineSpec,
        options: LoadOptions | None = None,
    ) -> None:
        self.model = model
        self.pipeline_spec = pipeline_spec
        self.options = options or LoadOptions()
        self._loaded = False

    # --- Availability ------------------------------------------------------ #

    @classmethod
    @abstractmethod
    def availability(cls) -> BackendAvailability:
        """Can this backend run here? Must not raise, and must not load weights.

        Imports of the runtime belong *inside* this method: importing
        ``imagegen.backends`` must never import torch.
        """

    @classmethod
    def supports_device(cls, device: Device) -> bool:
        return device.runtime in cls.runtimes

    # --- Lifecycle --------------------------------------------------------- #

    @abstractmethod
    def load(self, progress: ProgressCallback | None = None) -> None:
        """Fetch, instantiate and, where applicable, compile the model."""

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> list[GeneratedImage]:
        """Run inference. ``request`` is already defaulted and validated.

        Always returns PIL images, whatever the runtime hands back.
        """

    def unload(self) -> None:
        """Release device memory. Safe to call twice."""
        self._loaded = False

    def accepts(self, spec: PipelineSpec) -> bool:
        """Whether this loaded backend can serve ``spec`` without reloading.

        The default implementation consults
        :attr:`BackendCapabilities.recompiles_on`, so a dynamic backend accepts
        anything with the same model and device plan, while a static one accepts
        only an identical compile key.
        """
        if spec.model_key != self.pipeline_spec.model_key:
            return False
        if spec.device_plan != self.pipeline_spec.device_plan:
            return False
        # Always a reload, on every backend: an effort level can select a
        # different adapter, and the previous one was fused into the weights.
        if spec.effort is not self.pipeline_spec.effort:
            return False
        triggers = self.capabilities.recompiles_on
        if not triggers:
            return True
        for attribute in triggers:
            if getattr(spec, attribute, None) != getattr(self.pipeline_spec, attribute, None):
                return False
        return True

    # --- Introspection ----------------------------------------------------- #

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """What this backend + model + compile config supports.

        Must be usable *before* :meth:`load`, from registry metadata alone, so
        that an impossible request fails before a multi-gigabyte download.
        """

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> Device | None:
        plan = self.pipeline_spec.device_plan
        return plan.default if plan else None

    def memory_usage_bytes(self) -> int | None:
        """Device memory currently held, when the runtime can report it."""
        return None

    def can_afford_upscale(self, scale: float) -> tuple[bool, str]:
        """Whether a refinement pass at ``scale`` fits in device memory.

        The top of the effort ladder denoises an upscaled copy, which is the
        heaviest call of the run. On a card where it does not fit, the honest
        answer is to refine at the original size and say so - not to let the
        driver spill to system RAM and turn a 40-second generation into a
        ten-minute one.

        Returns ``(True, "")`` by default: a backend that cannot estimate its
        memory should not veto the feature.
        """
        return True, ""

    def inspect_request(self, request: GenerationRequest) -> list[str]:
        """Warnings that can only be found once the model is loaded.

        Separate from :meth:`validate`, which must work from registry metadata
        alone so that an impossible request fails before a multi-gigabyte
        download. Tokenizer limits and denoiser config are only knowable after.
        """
        return []

    # --- Shared request validation ----------------------------------------- #

    def validate(self, request: GenerationRequest) -> list[str]:
        """Check a request against :attr:`capabilities`.

        Returns warnings for parameters that will be ignored, and raises
        :class:`UnsupportedCapabilityError` for anything that would silently
        produce a wrong image.
        """
        caps = self.capabilities
        warnings: list[str] = []

        if request.mode is Mode.IMAGE_TO_IMAGE and not caps.image_to_image:
            missing = Component.VAE_ENCODER not in caps.components
            detail = (
                " (l'artefact ne contient pas d'encodeur VAE)"
                if missing
                else ""
            )
            raise UnsupportedCapabilityError(
                f"Le backend '{self.name}' avec le modele '{self.model.key}' ne fait pas "
                f"d'image-to-image{detail}.",
                hint="Retirez --image, ou choisissez un modele/backend qui le supporte "
                "(voir 'imagegen models').",
            )
        if request.mode is Mode.TEXT_TO_IMAGE and not caps.text_to_image:
            raise UnsupportedCapabilityError(
                f"Le backend '{self.name}' ne peut pas generer sans image source.",
                hint="Passez --image <fichier>.",
            )

        if request.negative_prompt and not caps.supports_negative_prompt:
            warnings.append(
                "Prompt negatif ignore : ce pipeline est compile sans guidage classifier-free "
                "(batch=1, une seule branche conditionnelle)."
            )
        if (
            request.guidance_scale is not None
            and request.guidance_scale > 1.0
            and not caps.supports_guidance
        ):
            warnings.append("guidance_scale ignore : ce pipeline est compile sans CFG.")
        if request.scheduler and request.scheduler not in caps.supported_schedulers:
            available = ", ".join(s.value for s in caps.supported_schedulers)
            warnings.append(
                f"Scheduler '{request.scheduler.value}' non disponible sur ce backend "
                f"(disponibles : {available}). Utilisation du scheduler par defaut."
            )
        if request.seed is not None and not caps.supports_seed:
            warnings.append("Graine ignoree : ce backend n'expose pas de RNG deterministe.")
        if request.num_images > 1 and not caps.supports_batch:
            warnings.append("Lot non supporte : les images seront produites une par une.")

        # img2img runs int(steps * strength) steps. Below 1 the input image comes
        # back untouched with no error - catch it here instead.
        if (
            request.mode is Mode.IMAGE_TO_IMAGE
            and request.steps is not None
            and request.effective_steps < 1
        ):
            needed = self.model.min_steps_for_strength(request.strength)
            raise UnsupportedCapabilityError(
                f"steps={request.steps} x strength={request.strength} donne "
                f"{request.effective_steps} etape de debruitage : l'image source serait "
                "renvoyee telle quelle.",
                hint=f"Utilisez --steps {needed} ou plus, ou augmentez --strength.",
            )

        # Resolution: on a static-shape backend an unsupported size must fail
        # loudly rather than be silently rounded.
        width, height = request.width, request.height
        if (
            width
            and height
            and not caps.dynamic_resolution
            and caps.fixed_resolutions
            and (width, height) not in caps.fixed_resolutions
        ):
            shapes = ", ".join(f"{w}x{h}" for w, h in caps.fixed_resolutions)
            raise UnsupportedCapabilityError(
                f"{width}x{height} n'est pas une resolution compilee pour ce backend.",
                hint=f"Resolutions disponibles : {shapes}. En changer impose de recompiler "
                "le modele (operation longue).",
            )
        multiple = caps.resolution_multiple
        for label, value in (("--width", width), ("--height", height)):
            if value is not None and multiple > 1 and value % multiple != 0:
                raise UnsupportedCapabilityError(
                    f"{label}={value} doit etre un multiple de {multiple} pour ce backend.",
                    hint=f"Essayez {max(multiple, round(value / multiple) * multiple)}.",
                )
        return warnings

    def __repr__(self) -> str:  # pragma: no cover - display helper
        state = "loaded" if self._loaded else "not loaded"
        return f"<{type(self).__name__} {self.pipeline_spec.describe()} {state}>"
