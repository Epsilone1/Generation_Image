"""ONNX Runtime backend: the common substrate for non-Intel NPUs and DX12 GPUs.

**Status: implemented, unvalidated.** The generic ORT path (CPU, DirectML, CUDA
EP) follows documented ``optimum-onnx`` APIs. The NPU execution providers cannot
be exercised here - QNN needs ARM64 Snapdragon hardware, VitisAI needs an AMD
XDNA part and AMD's Ryzen AI SDK - so those report themselves as experimental
instead of pretending to work.

The facts this backend is built around:

* **One ONNX Runtime distribution per environment.** ``onnxruntime``,
  ``-gpu``, ``-directml``, ``-qnn``, ``-openvino`` and ``-windowsml`` all claim
  the ``onnxruntime`` module name, so the flavour is an *install-time* choice
  exposed as a pip extra. Detection must therefore distinguish "built without
  provider X" from "hardware absent" - conflating them sends users to buy
  hardware they already have.

* **QNN and VitisAI are absolutely static.** The ORT documentation states that
  the QNN provider "does not support models with dynamic shapes"; every
  dimension is frozen at compile time, including the 77-token sequence length
  and the CFG batch factor. AMD's newer bundles relax this to an *enumerable set*
  of resolutions rather than a single one - which is why
  :attr:`BackendCapabilities.fixed_resolutions` is a tuple.

* **img2img can be missing for a given artefact.** ``optimum-cli export onnx``
  emits ``vae_encoder/`` only when exporting the img2img or inpaint task, and
  Qualcomm's published Stable Diffusion bundles ship text encoder, UNet and VAE
  *decoder* only. So img2img availability is read off the artefact's components,
  never assumed from the backend.

* **No LoRA, no arbitrary checkpoints** on the pre-compiled NPU paths: what runs
  is exactly what the vendor compiled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from ..errors import BackendNotImplementedError, ModelLoadError
from ..hardware.device import Runtime
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


#: Providers whose graphs are compiled per shape.
_STATIC_PROVIDERS = frozenset(
    {"QNNExecutionProvider", "VitisAIExecutionProvider", "TensorrtExecutionProvider",
     "NvTensorRtRtxExecutionProvider"}
)

#: Providers that run pre-compiled vendor bundles only: no LoRA, no arbitrary
#: checkpoints, and a hard allow-list of models.
_VENDOR_LOCKED_PROVIDERS = frozenset({"QNNExecutionProvider", "VitisAIExecutionProvider"})

#: Which pip extra provides which provider, for actionable install hints.
_PROVIDER_PACKAGES: dict[str, str] = {
    "QNNExecutionProvider": 'uv pip install "imagegen[onnx-qnn]"  (Snapdragon X, Windows ARM64)',
    "VitisAIExecutionProvider": "installez le Ryzen AI Software SDK d'AMD (non distribue sur PyPI)",
    "DmlExecutionProvider": 'uv pip install "imagegen[onnx-directml]"',
    "OpenVINOExecutionProvider": "uv pip install onnxruntime-openvino",
    "CUDAExecutionProvider": "uv pip install onnxruntime-gpu",
}


class OnnxRuntimeBackend(Backend):
    """Runs ONNX diffusion graphs through an ONNX Runtime execution provider."""

    name: ClassVar[str] = "onnx"
    runtimes: ClassVar[tuple[Runtime, ...]] = (Runtime.ONNXRUNTIME,)
    priority: ClassVar[int] = 50
    model_format: ClassVar[str] = "onnx"
    description: ClassVar[str] = "ONNX Runtime (DirectML / QNN / VitisAI / CUDA) - experimental"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pipeline: Any = None
        self._components: frozenset[Component] | None = None

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #

    @classmethod
    def availability(cls) -> BackendAvailability:
        try:
            import onnxruntime as ort
        except ImportError:
            return BackendAvailability(
                False,
                "'onnxruntime' n'est pas installe",
                install_hint='uv pip install "imagegen[onnx-directml]"  '
                "(ou [onnx-qnn] sur Snapdragon)",
            )
        try:
            import optimum.onnxruntime  # noqa: F401
        except ImportError:
            return BackendAvailability(
                False,
                "'optimum[onnxruntime]' n'est pas installe (onnxruntime seul ne fournit pas "
                "les pipelines de diffusion)",
                install_hint='uv pip install "imagegen[onnx]"',
            )

        providers = [p for p in ort.get_available_providers() if p not in {"CPUExecutionProvider", "AzureExecutionProvider"}]
        if not providers:
            # The distinction that matters: the package is here, the provider is
            # not compiled into it. Telling the user "no NPU found" when they own
            # one sends them looking in the wrong place entirely.
            return BackendAvailability(
                False,
                f"onnxruntime {ort.__version__} est compile sans provider accelere "
                f"(seul le CPU est disponible)",
                install_hint="Une seule distribution onnxruntime peut etre installee a la fois. "
                "Choisissez celle de votre materiel : onnxruntime-directml, onnxruntime-qnn, "
                "onnxruntime-gpu, onnxruntime-openvino.",
            )

        experimental = any(p in _VENDOR_LOCKED_PROVIDERS for p in providers)
        return BackendAvailability(
            True,
            reason=("providers NPU non valides sur materiel" if experimental else ""),
            implemented=not experimental,
        )

    # ------------------------------------------------------------------ #
    # Capabilities
    # ------------------------------------------------------------------ #

    @property
    def capabilities(self) -> BackendCapabilities:
        device = self.device
        provider = device.handle if device else ""
        static = provider in _STATIC_PROVIDERS
        locked = provider in _VENDOR_LOCKED_PROVIDERS

        if self._components is not None:
            components = self._components
        else:
            components = frozenset(
                {
                    Component.TOKENIZER,
                    Component.TEXT_ENCODER,
                    Component.DENOISER,
                    Component.VAE_DECODER,
                    Component.SCHEDULER,
                }
                | (
                    {Component.VAE_ENCODER}
                    if self.model.img2img_mode is not Img2ImgMode.UNSUPPORTED and not locked
                    else set()
                )
            )

        fixed: tuple[tuple[int, int], ...] = ()
        if static:
            # Vendor bundles publish an allow-list; 512x512 is the shape every
            # published Stable Diffusion bundle ships.
            fixed = ((512, 512),) if locked else ((self.pipeline_spec.width, self.pipeline_spec.height),)

        return BackendCapabilities(
            components=components,
            # Only if the compiled UNet was built with the doubled CFG batch.
            # Vendor bundles frequently compile batch=1 to fit NPU memory, which
            # makes the negative prompt a silent no-op.
            supports_negative_prompt=not locked,
            supports_guidance=not locked,
            supports_seed=True,
            deterministic_seed=False,  # int8/w8a16 weights change the output
            supports_batch=not static,
            supports_progress_callback=True,
            supports_cancel=False,
            dynamic_resolution=not static,
            fixed_resolutions=fixed,
            resolution_multiple=64 if static else 8,
            recompiles_on=frozenset(
                {"width", "height", "batch_size", "guidance_scale"} if static else set()
            ),
            supports_compiled_cache=True,
            supported_schedulers=(
                SchedulerKind.AUTO,
                SchedulerKind.EULER,
                SchedulerKind.EULER_ANCESTRAL,
                SchedulerKind.DDIM,
                SchedulerKind.LCM,
            ),
            supported_precisions=(Precision.FP32, Precision.FP16, Precision.INT8),
            supports_lora=False,  # a LoRA must be baked in before conversion
            supports_controlnet=False,
            supports_ip_adapter=False,
            supports_inpainting=False,
        )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _artifact(self) -> str:
        portable = self.model.portable
        if portable.onnx_repo and portable.onnx_repo != "in-repo":
            return portable.onnx_repo
        if portable.onnx_repo == "in-repo":
            return self.model.repo_id
        raise ModelLoadError(
            f"Aucun graphe ONNX connu pour le modele '{self.model.key}'.",
            hint="Exportez-le d'abord :\n"
            f"  optimum-cli export onnx --model {self.model.repo_id} --task image-to-image "
            f"{self.model.key}-onnx/\n"
            "Le sous-dossier vae_encoder/ n'est produit que pour la tache image-to-image : "
            "sans lui, l'img2img sera definitivement indisponible pour cet artefact.",
        )

    def load(self, progress: ProgressCallback | None = None) -> None:
        availability = self.availability()
        availability.raise_if_unavailable(self.name)

        device = self.device
        if device is None:  # pragma: no cover
            raise ModelLoadError("Aucun provider ONNX Runtime selectionne.")

        if device.handle in _VENDOR_LOCKED_PROVIDERS:
            raise BackendNotImplementedError(
                f"Le provider {device.handle} n'execute que des artefacts pre-compiles par le "
                "constructeur, non couverts par ce registre.",
                hint=_PROVIDER_PACKAGES.get(device.handle, "")
                + "\nLes modeles compiles se recuperent chez le constructeur "
                "(Qualcomm AI Hub, AMD Ryzen AI), et la liste des modeles supportes fait foi.",
            )

        from optimum.onnxruntime import ORTDiffusionPipeline  # type: ignore[import-not-found]

        artifact = self._artifact()
        try:
            self._pipeline = ORTDiffusionPipeline.from_pretrained(
                artifact,
                provider=device.handle,
                cache_dir=self.options.cache_dir,
                local_files_only=self.options.local_files_only,
                token=self.options.token,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Chargement du graphe ONNX '{artifact}' impossible : {exc}"
            ) from exc

        self._components = _detect_components(self._pipeline, artifact)
        self._loaded = True

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

        import numpy as np

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
            # ORT pipelines run their scheduler in numpy on the CPU.
            "generator": np.random.RandomState(seed),
        }
        if request.negative_prompt and self.capabilities.supports_negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.mode is Mode.IMAGE_TO_IMAGE:
            kwargs["image"] = request.init_image
            kwargs["strength"] = request.strength
        else:
            kwargs["width"] = self.pipeline_spec.width
            kwargs["height"] = self.pipeline_spec.height

        try:
            output = self._pipeline(**kwargs)
        except Exception as exc:
            raise ModelLoadError(f"Generation ONNX Runtime en echec : {exc}") from exc

        parameters = {
            "model": self.model.key,
            "backend": self.name,
            "device": self.device.short_id if self.device else "?",
            "steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
        }
        return [
            GeneratedImage(image=image, seed=seed + index, index=index, parameters=dict(parameters))
            for index, image in enumerate(output.images)
        ]

    def unload(self) -> None:
        self._pipeline = None
        self._components = None
        self._loaded = False


def _detect_components(pipeline: Any, artifact: str) -> frozenset[Component]:
    """Read the artefact's real components.

    An ONNX export without ``vae_encoder/`` can never do img2img, no matter what
    the model registry says about the family. This is the check that turns a
    confusing wrong image into a clear refusal.
    """
    found = {
        Component.TOKENIZER,
        Component.TEXT_ENCODER,
        Component.DENOISER,
        Component.VAE_DECODER,
        Component.SCHEDULER,
    }
    encoder = getattr(pipeline, "vae_encoder", None)
    if encoder is not None:
        found.add(Component.VAE_ENCODER)
    else:
        local = Path(artifact)
        if local.is_dir() and (local / "vae_encoder").is_dir():
            found.add(Component.VAE_ENCODER)
    if getattr(pipeline, "text_encoder_2", None) is not None:
        found.add(Component.TEXT_ENCODER_2)
    return frozenset(found)
