"""The facade: turns a request into images, choosing model, device and backend.

This is the only object a caller needs. It owns three things the backends
deliberately do not:

* **Defaulting.** A preset's steps, guidance, scheduler and resolution are
  merged into the request here, so a backend never guesses and the recorded
  parameters are exactly what ran. ``guidance_scale`` defaults to ``None`` and is
  resolved per model rather than to a constant - a CLI default of 7.5 leaking
  into a distilled model produces burnt, oversaturated output with no error.

* **The pipeline cache.** Loading is expensive and, on NPU backends,
  *recompiling* is expensive. Requests are turned into a
  :class:`~imagegen.types.PipelineSpec`, and a loaded backend is reused whenever
  it :meth:`~imagegen.backends.base.Backend.accepts` the new one. When it does
  not, the reload is reported as its own timing rather than hidden inside the
  generation.

* **Image sizing.** img2img takes its output size from the source image, so
  ``--width/--height`` has to be applied by resizing the input before the
  backend ever sees it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..backends import Backend, LoadOptions, select_backend_class
from ..errors import ConfigurationError
from ..hardware.detect import select_device
from ..hardware.device import Device
from ..models.registry import (
    DEFAULT_MODEL_KEY,
    EffortProfile,
    ModelSpec,
    get_model,
    recommend_model,
)
from ..types import (
    DEFAULT_EFFORT,
    CancelCallback,
    DevicePlan,
    Effort,
    GeneratedImage,
    GenerationRequest,
    GenerationResult,
    Mode,
    PipelineSpec,
    Precision,
    ProgressCallback,
    SchedulerKind,
    Stopwatch,
)
from .images import fit_dimensions, load_image, prepare_init_image, round_to_multiple
from .prompts import PreparedPrompt, PromptTranslator, prepare_prompts

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResolvedPlan:
    """What a request resolves to, before anything is loaded.

    Carries its own warnings so that an adjustment the generator made on the
    user's behalf - raising the step count so that img2img actually denoises,
    for instance - is reported rather than applied silently.
    """

    spec: PipelineSpec
    request: GenerationRequest
    warnings: list[str] = field(default_factory=list)
    #: The prompt as it will reach the encoders, and what it came from.
    prompt: PreparedPrompt | None = None


class ImageGenerator:
    """Generates images from a prompt and, optionally, a source image."""

    def __init__(
        self,
        model: ModelSpec,
        device: Device,
        backend_class: type[Backend],
        options: LoadOptions | None = None,
        *,
        precision: Precision = Precision.AUTO,
        effort: Effort = DEFAULT_EFFORT,
        translate: str = "auto",
    ) -> None:
        self.model = model
        self.device = device
        self.backend_class = backend_class
        self.options = options or LoadOptions()
        self.precision = precision
        self.effort = effort
        self.translate = translate
        self._backend: Backend | None = None
        self._last_load_seconds = 0.0
        self._translator = PromptTranslator(
            cache_dir=self.options.cache_dir,
            local_files_only=self.options.local_files_only,
        )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        model: str | None = None,
        device: str = "auto",
        backend: str | None = None,
        options: LoadOptions | None = None,
        *,
        precision: Precision = Precision.AUTO,
        effort: Effort = DEFAULT_EFFORT,
        translate: str = "auto",
    ) -> ImageGenerator:
        """Resolve model, device and backend without loading any weights.

        Nothing is downloaded here, so an impossible combination fails in
        milliseconds instead of after several gigabytes.
        """
        selected_device = select_device(device)
        if model:
            spec = get_model(model)
        else:
            # An 8 GB discrete GPU and a memory-sharing NPU deserve different
            # defaults; the registry decides from the reported memory.
            spec = recommend_model(selected_device.total_memory_gb)
            logger.debug("Modele choisi automatiquement : %s", spec.key)

        backend_class = select_backend_class(selected_device, backend)
        return cls(
            spec,
            selected_device,
            backend_class,
            options,
            precision=precision,
            effort=effort,
            translate=translate,
        )

    @classmethod
    def for_model(cls, model_key: str = DEFAULT_MODEL_KEY, **kwargs: Any) -> ImageGenerator:
        return cls.create(model=model_key, **kwargs)

    # ------------------------------------------------------------------ #
    # Spec building
    # ------------------------------------------------------------------ #

    def build_spec(self, request: GenerationRequest) -> ResolvedPlan:
        """Derive the compile key and the fully-defaulted request.

        Returned together because resolving one settles the other: the output
        resolution determines the spec, and the spec's guidance value is what the
        request will actually run with.
        """
        model = self.model
        warnings: list[str] = []
        profile = model.effort(self.effort)

        # Explicit flags win over the effort level: the level exists to fill in
        # what the user did not pin, not to override what they did.
        guidance = request.guidance_scale
        if guidance is None:
            guidance = profile.guidance if profile.guidance is not None else model.default_guidance
        steps = request.steps if request.steps is not None else profile.steps
        scheduler = request.scheduler or SchedulerKind.AUTO

        width, height = self._resolve_size(request)

        # The text encoders are English-trained: a French prompt does not fail,
        # it quietly drops the words CLIP cannot represent. Translating is done
        # here, before anything else sees the request, and always reported.
        prompt, negative = prepare_prompts(
            request.prompt,
            request.negative_prompt,
            mode=self.translate,
            translator=self._translator,
        )
        if prompt.note:
            warnings.append(prompt.note)

        # img2img runs int(steps * strength) denoising steps; below one step the
        # source image is returned untouched with no error from diffusers.
        if request.mode is Mode.IMAGE_TO_IMAGE:
            steps = self._raise_steps_for_strength(steps, request.strength, warnings)

        resolved = GenerationRequest(
            prompt=prompt.text,
            negative_prompt=negative.text if negative else None,
            init_image=request.init_image,
            strength=request.strength,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance,
            scheduler=scheduler,
            seed=request.seed,
            num_images=request.num_images,
            extra=dict(request.extra),
        )

        spec = PipelineSpec(
            model_key=model.key,
            mode=request.mode,
            width=width,
            height=height,
            batch_size=request.num_images,
            guidance_scale=guidance,
            precision=self.precision,
            scheduler=scheduler,
            effort=self.effort,
            device_plan=DevicePlan(self.device),
        )
        return ResolvedPlan(spec, resolved, warnings, prompt)

    def _resolve_size(self, request: GenerationRequest) -> tuple[int, int]:
        """Decide the output resolution.

        Explicit flags win. Otherwise a text-to-image request uses the model's
        native size, and an image-to-image request follows the source image's
        aspect ratio scaled to the model's native pixel count.
        """
        model = self.model
        if request.width and request.height:
            return request.width, request.height

        if request.mode is Mode.TEXT_TO_IMAGE:
            width = request.width or model.default_width
            height = request.height or model.default_height
            return width, height

        image = self._as_image(request.init_image)
        native_pixels = model.default_width * model.default_height
        width, height = fit_dimensions(image.width, image.height, native_pixels, multiple=8)
        if request.width:
            width = request.width
        if request.height:
            height = request.height
        return width, height

    def _raise_steps_for_strength(
        self, steps: int, strength: float, warnings: list[str]
    ) -> int:
        """Ensure at least one denoising step actually runs.

        Corrected rather than refused: the user asked for something impossible
        and clearly meant "as few steps as this strength allows". But the
        correction changes a value they set explicitly, so it is reported.
        """
        if strength <= 0:
            raise ConfigurationError("strength doit etre superieur a 0 pour l'image-to-image.")
        effective = min(int(steps * strength), steps)
        if effective >= 1:
            return steps
        needed = self.model.min_steps_for_strength(strength)
        warnings.append(
            f"--steps {steps} avec --strength {strength:g} ne lance aucune etape de debruitage "
            f"(int({steps} x {strength:g}) = 0) : --steps porte a {needed}."
        )
        return needed

    @staticmethod
    def _as_image(source: Path | Image | None) -> Image:
        if source is None:  # pragma: no cover - guarded by Mode
            raise ConfigurationError("Aucune image source fournie.")
        if isinstance(source, (str, Path)):
            return load_image(source)
        return source

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def ensure_backend(
        self, spec: PipelineSpec, progress: ProgressCallback | None = None
    ) -> tuple[Backend, bool]:
        """Return a backend ready for ``spec``, loading or reloading as needed.

        The boolean says whether a load happened, so the caller can report a
        multi-minute NPU recompile as such instead of as a slow generation.
        """
        backend = self._backend
        if backend is not None and backend.is_loaded and backend.accepts(spec):
            return backend, False

        if backend is not None:
            logger.debug("Specification incompatible avec le pipeline charge : rechargement.")
            backend.unload()

        # The memory plan has to know the largest canvas the run will touch: a
        # refinement pass denoises an upscaled copy, and planning from the base
        # resolution alone would under-provision its heaviest call.
        profile = self.model.effort(spec.effort)
        options = replace(
            self.options,
            peak_resolution_scale=profile.refine_scale if profile.refines else 1.0,
        )
        backend = self.backend_class(self.model, spec, options)
        # Validated before the download: a request this backend cannot serve
        # should not cost several gigabytes to discover.
        with Stopwatch() as watch:
            backend.load(progress)
        self._last_load_seconds = watch.elapsed
        self._backend = backend
        return backend, True

    @property
    def backend(self) -> Backend | None:
        return self._backend

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
        load_progress: ProgressCallback | None = None,
        plan: ResolvedPlan | None = None,
    ) -> GenerationResult:
        """Produce images for ``request``.

        ``plan`` lets a caller that already resolved the request - the CLI does,
        to print it before starting - reuse that work instead of translating and
        defaulting a second time.
        """
        plan = plan or self.build_spec(request)
        spec, resolved = plan.spec, plan.request

        # Capabilities are knowable from registry metadata alone, so an
        # unsupported request fails before any weights are fetched.
        probe = self.backend_class(self.model, spec, self.options)
        warnings = [*plan.warnings, *probe.validate(resolved)]

        backend, loaded = self.ensure_backend(spec, load_progress)
        warnings.extend(backend.validate(resolved))
        # Tokenizer limits and denoiser config need the loaded model.
        warnings.extend(backend.inspect_request(resolved))

        memory = getattr(backend, "memory_decision", None)
        if memory is not None:
            warnings.extend(memory.warnings)

        if resolved.mode is Mode.IMAGE_TO_IMAGE:
            image = self._as_image(resolved.init_image)
            resolved.init_image = prepare_init_image(image, spec.width, spec.height)

        profile = self.model.effort(self.effort)
        with Stopwatch() as watch:
            images: list[GeneratedImage] = backend.generate(resolved, progress, cancel)
            if profile.refines:
                images = self._refine(backend, resolved, images, profile, progress, cancel, warnings)

        # The recorded prompt is what the encoders saw; the original is what the
        # user typed. An image reproducible only from the translation would be a
        # trap the day the translation model changes.
        if plan.prompt is not None and plan.prompt.translated:
            for generated in images:
                generated.parameters["prompt_original"] = plan.prompt.original
                generated.parameters["prompt_language"] = plan.prompt.source_language
        generated_effort = {"effort": self.effort.value}
        for generated in images:
            generated.parameters.update(generated_effort)

        return GenerationResult(
            images=images,
            mode=resolved.mode,
            model_key=self.model.key,
            backend=backend.name,
            device=self.device.short_id,
            duration_s=watch.elapsed,
            load_s=self._last_load_seconds if loaded else 0.0,
            warnings=_dedupe(warnings),
        )

    def _refine(
        self,
        backend: Backend,
        request: GenerationRequest,
        images: list[GeneratedImage],
        profile: EffortProfile,
        progress: ProgressCallback | None,
        cancel: CancelCallback | None,
        warnings: list[str],
    ) -> list[GeneratedImage]:
        """Run a second, low-strength image-to-image pass over each result.

        This is the top of the effort ladder for models whose step count is
        fixed: a distilled checkpoint cannot simply be run longer, but its output
        can be re-denoised from a small amount of added noise, which recovers
        the fine texture few-step sampling leaves out.

        It needs a VAE encoder, so it is skipped - with a warning, never
        silently - on a backend or artefact that has none.
        """
        if not backend.capabilities.image_to_image:
            warnings.append(
                f"Passe d'affinage ignoree : le backend '{backend.name}' ne fait pas d'img2img "
                "avec ce modele."
            )
            return images

        scale = profile.refine_scale
        if scale > 1.0:
            # Upscaling makes this the heaviest call of the run. Where it does
            # not fit, refine at the original size and say so: on Windows an
            # over-capacity call does not raise, it spills to system RAM and
            # takes ten times longer.
            affordable, reason = backend.can_afford_upscale(scale)
            if not affordable:
                warnings.append(
                    f"Affinage ramene a la resolution d'origine : {reason}."
                )
                scale = 1.0

        target = None
        if scale > 1.0:
            # Snap to a multiple of 64: SDXL's VAE works at 1/8 and its UNet
            # downsamples three more times, so odd sizes get padded badly.
            target = (
                round_to_multiple(int(request.width * scale), 64),
                round_to_multiple(int(request.height * scale), 64),
            )

        refined: list[GeneratedImage] = []
        for generated in images:
            source = generated.image
            if target is not None:
                source = prepare_init_image(source, *target)
            pass_request = replace(
                request,
                init_image=source,
                strength=profile.refine_strength,
                steps=profile.refine_steps,
                width=target[0] if target else request.width,
                height=target[1] if target else request.height,
                seed=generated.seed,
                num_images=1,
            )
            result = backend.generate(pass_request, progress, cancel)
            if not result:  # pragma: no cover - defensive
                refined.append(generated)
                continue
            improved = result[0]
            refined.append(
                GeneratedImage(
                    image=improved.image,
                    seed=generated.seed,
                    index=generated.index,
                    parameters={
                        **generated.parameters,
                        "refine_strength": profile.refine_strength,
                        "refine_steps": profile.refine_steps,
                        "refine_scale": scale,
                        "final_size": list(improved.image.size),
                    },
                )
            )
        return refined

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None

    def __enter__(self) -> ImageGenerator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def describe(self) -> dict[str, Any]:
        """Everything resolved, for ``imagegen info`` and ``--json``."""
        return {
            "model": self.model.key,
            "repo_id": self.model.repo_id,
            "backend": self.backend_class.name,
            "device": self.device.short_id,
            "device_name": self.device.name,
            "device_kind": self.device.kind.value,
            "precision": self.precision.value,
            "license": self.model.license.name,
            "commercial_use": self.model.license.commercial.value,
            "supports_img2img": self.model.supports_img2img,
        }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


__all__ = ["ImageGenerator", "ResolvedPlan"]
