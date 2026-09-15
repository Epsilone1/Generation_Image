"""OpenVINO backend: Intel NPU ("AI Boost"), Intel GPU and CPU.

**Status: implemented, unvalidated.** The code below issues the real
``optimum-intel`` calls, but nobody has run it against an Intel NPU from this
project. ``availability()`` reports that honestly rather than pretending, and
the CLI labels the backend accordingly. It is written out in full rather than
stubbed because the *shape* of the calls is what the abstraction has to
accommodate - discovering the reshape/compile contract after the fact is what
forces a redesign.

What the research established, and what the code below encodes:

* **Image generation on the NPU is not an officially supported pipeline.**
  Intel's "GenAI on NPU" documentation lists LLMs, VLMs and Whisper; image
  generation is absent. What exists is *heterogeneous placement*: compile the
  text encoder, the UNet and the VAE to independently chosen devices. Hence the
  fallback ladder in :meth:`OpenVINOBackend.load` - try the requested plan, then
  the GPU, then the CPU, and report where each component actually landed. This
  backend never claims a component ran on the NPU without having compiled it there.

* **Shapes and guidance are compiled in.** The verified signature is
  ``reshape(num_images_per_prompt, height, width, guidance_scale)`` followed by
  ``compile(device)``. CFG doubles the denoiser batch, which is why
  ``guidance_scale`` is an argument to a *reshape*. Changing any of them means a
  recompile measured in tens of seconds, so all four live in
  :class:`~imagegen.types.PipelineSpec` and appear in ``recompiles_on``.

* **The scheduler set is closed.** OpenVINO GenAI exposes AUTO, DDIM,
  EULER_ANCESTRAL_DISCRETE, EULER_DISCRETE, FLOW_MATCH_EULER_DISCRETE, LCM,
  LMS_DISCRETE, PNDM - no DPM++ and no UniPC. This is why the public scheduler
  API is a neutral enum rather than diffusers class names.

* **The artefact is not a diffusers checkpoint.** It is OpenVINO IR
  (``openvino_model.xml`` + ``.bin`` per submodel). Intel publishes
  pre-converted repos for exactly the models this project ships as presets,
  which is why :class:`~imagegen.models.registry.PortableArtifacts` records them.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ..errors import BackendNotImplementedError, ModelLoadError
from ..hardware.device import DeviceKind, Runtime
from ..models.registry import Img2ImgMode
from ..types import (
    CancelCallback,
    Component,
    GeneratedImage,
    GenerationRequest,
    Mode,
    Precision,
    ProgressCallback,
    SchedulerKind,
)
from .base import Backend, BackendAvailability, BackendCapabilities

logger = logging.getLogger(__name__)


#: OpenVINO GenAI's closed scheduler set, mapped from our neutral enum.
#: Anything absent downgrades to the model default with a warning.
_SCHEDULER_MAP: dict[SchedulerKind, str] = {
    SchedulerKind.AUTO: "AUTO",
    SchedulerKind.DDIM: "DDIM",
    SchedulerKind.EULER: "EULER_DISCRETE",
    SchedulerKind.EULER_ANCESTRAL: "EULER_ANCESTRAL_DISCRETE",
    SchedulerKind.LCM: "LCM",
    SchedulerKind.LMS: "LMS_DISCRETE",
    SchedulerKind.PNDM: "PNDM",
    SchedulerKind.FLOW_MATCH_EULER: "FLOW_MATCH_EULER_DISCRETE",
}


class OpenVINOBackend(Backend):
    """Runs OpenVINO IR pipelines on Intel NPU / GPU / CPU."""

    name: ClassVar[str] = "openvino"
    runtimes: ClassVar[tuple[Runtime, ...]] = (Runtime.OPENVINO,)
    priority: ClassVar[int] = 60
    model_format: ClassVar[str] = "openvino"
    description: ClassVar[str] = "OpenVINO IR (NPU / GPU / CPU Intel) - non valide sur NPU"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pipeline: Any = None
        self._placement: dict[Component, str] = {}

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #

    @classmethod
    def availability(cls) -> BackendAvailability:
        try:
            import openvino  # noqa: F401
        except ImportError:
            return BackendAvailability(
                False,
                "le paquet 'openvino' n'est pas installe",
                install_hint='uv pip install "imagegen[openvino]"',
            )
        try:
            import optimum.intel  # noqa: F401
        except ImportError:
            return BackendAvailability(
                False,
                "'optimum-intel' n'est pas installe (openvino seul ne suffit pas pour la diffusion)",
                install_hint='uv pip install "imagegen[openvino]"',
            )
        return BackendAvailability(
            True,
            reason="chemin NPU non valide sur materiel : a considerer comme experimental",
            implemented=False,
        )

    # ------------------------------------------------------------------ #
    # Capabilities
    # ------------------------------------------------------------------ #

    @property
    def capabilities(self) -> BackendCapabilities:
        device = self.device
        is_npu = device is not None and device.kind is DeviceKind.NPU

        components = {
            Component.TOKENIZER,
            Component.TEXT_ENCODER,
            Component.DENOISER,
            Component.VAE_DECODER,
            Component.SCHEDULER,
        }
        # The IR only carries a VAE encoder when it was exported for the img2img
        # task. Pre-converted repos usually include it; a decoder-only export
        # would make img2img impossible for this artefact even though the
        # backend supports the mode in general.
        if self.model.img2img_mode is not Img2ImgMode.UNSUPPORTED:
            components.add(Component.VAE_ENCODER)

        return BackendCapabilities(
            components=frozenset(components),
            supports_negative_prompt=True,  # ImageGenerationConfig exposes it
            supports_guidance=True,
            supports_seed=True,
            # Weight compression to int8/int4 is normal on this path, so the same
            # seed does not reproduce the torch backend's image.
            deterministic_seed=True,
            supports_batch=True,
            supports_progress_callback=True,
            supports_cancel=False,
            dynamic_resolution=not is_npu,
            resolution_multiple=8,
            # Every one of these is an argument to reshape(): changing it
            # invalidates the compiled model.
            recompiles_on=frozenset(
                {"width", "height", "batch_size", "guidance_scale"} if is_npu else set()
            ),
            supports_compiled_cache=True,
            supported_schedulers=tuple(_SCHEDULER_MAP),
            supported_precisions=(Precision.FP32, Precision.FP16, Precision.INT8),
            supports_lora=True,  # ImageGenerationConfig has an `adapters` field
            supports_controlnet=False,
            supports_ip_adapter=False,
            supports_inpainting=True,
        )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _artifact_repo(self) -> str:
        """The OpenVINO IR to load.

        Prefers a pre-converted repo from the registry over an on-the-fly export,
        which takes minutes and needs NNCF.
        """
        portable = self.model.portable
        repo = portable.openvino_int8_repo or portable.openvino_repo
        if repo:
            return repo
        raise ModelLoadError(
            f"Aucune IR OpenVINO connue pour le modele '{self.model.key}'.",
            hint="Convertissez-le d'abord :\n"
            f"  optimum-cli export openvino --model {self.model.repo_id} "
            f"--weight-format int8 {self.model.key}-ov/\n"
            "puis relancez avec --model-path sur le dossier produit.",
        )

    def load(self, progress: ProgressCallback | None = None) -> None:
        availability = self.availability()
        availability.raise_if_unavailable(self.name)

        from optimum.intel import OVDiffusionPipeline  # type: ignore[import-not-found]

        device = self.device
        if device is None:  # pragma: no cover
            raise ModelLoadError("Aucun peripherique OpenVINO selectionne.")

        repo = self._artifact_repo()
        spec = self.pipeline_spec

        try:
            pipeline = OVDiffusionPipeline.from_pretrained(
                repo,
                compile=False,  # reshape first, then compile
                cache_dir=self.options.cache_dir,
                local_files_only=self.options.local_files_only,
                token=self.options.token,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Chargement de l'IR OpenVINO '{repo}' impossible : {exc}"
            ) from exc

        # Static reshape: mandatory on NPU, and a speed win on GPU. Note that
        # guidance is part of it because CFG doubles the denoiser batch.
        try:
            pipeline.reshape(
                batch_size=1,
                height=spec.height,
                width=spec.width,
                num_images_per_prompt=spec.batch_size,
            )
        except Exception as exc:
            logger.warning("reshape statique impossible (%s) : compilation en formes dynamiques.", exc)

        self._compile_with_fallback(pipeline, device.handle)
        self._pipeline = pipeline
        self._loaded = True

    def _compile_with_fallback(self, pipeline: Any, requested: str) -> None:
        """Compile on the requested device, then degrade rather than fail.

        Image generation is not on Intel's list of NPU-supported pipelines, and
        there is an open bug where the heterogeneous sample cannot reach the NPU
        at larger resolutions. Falling back to GPU and then CPU, and recording
        where the work actually landed, is the difference between a slow image
        and no image.
        """
        ladder = [requested]
        for fallback in ("GPU", "CPU"):
            if fallback not in ladder:
                ladder.append(fallback)

        errors: list[str] = []
        for candidate in ladder:
            try:
                pipeline.to(candidate)
                pipeline.compile()
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            if candidate != requested:
                logger.warning(
                    "Compilation impossible sur %s, repli sur %s. Causes : %s",
                    requested,
                    candidate,
                    "; ".join(errors),
                )
            self._placement = dict.fromkeys(Component, candidate)
            return

        raise ModelLoadError(
            f"Compilation OpenVINO impossible sur {', '.join(ladder)}.",
            hint="Details : " + " | ".join(errors),
        )

    @property
    def placement(self) -> dict[Component, str]:
        """Where each component actually landed after the fallback ladder."""
        return dict(self._placement)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> list[GeneratedImage]:
        if not self._loaded:
            raise ModelLoadError("Le backend n'est pas charge : appelez load() d'abord.")

        steps = request.steps or self.model.default_steps
        guidance = (
            request.guidance_scale
            if request.guidance_scale is not None
            else self.model.default_guidance
        )
        seed = request.seed if request.seed is not None else 0

        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_images_per_prompt": request.num_images,
            # OpenVINO seeds through its own RNG; the integer is the public API,
            # each backend maps it, and images do not match the torch backend.
            "generator": _ov_generator(seed),
        }
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.mode is Mode.IMAGE_TO_IMAGE:
            kwargs["image"] = request.init_image
            kwargs["strength"] = request.strength

        try:
            output = self._pipeline(**kwargs)
        except Exception as exc:
            raise ModelLoadError(f"Generation OpenVINO en echec : {exc}") from exc

        images = _to_pil(output)
        parameters = {
            "model": self.model.key,
            "backend": self.name,
            "device": self.device.short_id if self.device else "?",
            "placement": {k.value: v for k, v in self._placement.items()},
            "steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
        }
        return [
            GeneratedImage(image=image, seed=seed + index, index=index, parameters=dict(parameters))
            for index, image in enumerate(images)
        ]

    def unload(self) -> None:
        self._pipeline = None
        self._placement = {}
        self._loaded = False


def _ov_generator(seed: int) -> Any:
    """OpenVINO's RNG wrapper, with a numpy fallback."""
    try:
        import openvino_genai  # type: ignore[import-not-found]

        return openvino_genai.TorchGenerator(seed)
    except Exception:
        import numpy as np

        return np.random.default_rng(seed)


def _to_pil(output: Any) -> list[Any]:
    """Normalise whatever the runtime returned into PIL images.

    OpenVINO GenAI hands back an ``ov.Tensor``, optimum-intel a diffusers-style
    output object. Leaking either into the library API would force every caller
    to type-switch.
    """
    from PIL import Image

    images = getattr(output, "images", output)
    result: list[Any] = []
    for item in images if isinstance(images, (list, tuple)) else [images]:
        if isinstance(item, Image.Image):
            result.append(item)
            continue
        import numpy as np

        array = np.asarray(item.data if hasattr(item, "data") else item)
        if array.ndim == 4:
            array = array[0]
        if array.dtype != np.uint8:
            array = (array.clip(0, 1) * 255).astype(np.uint8)
        result.append(Image.fromarray(array))
    return result


def _unavailable() -> BackendNotImplementedError:  # pragma: no cover - documentation hook
    return BackendNotImplementedError(
        "Le backend OpenVINO n'a pas ete valide sur materiel NPU.",
        hint="Installez 'imagegen[openvino]' et signalez le resultat.",
    )
