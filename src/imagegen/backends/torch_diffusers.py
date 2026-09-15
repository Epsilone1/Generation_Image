"""PyTorch + diffusers backend: the reference implementation.

This is the only fully implemented backend. It drives CUDA, Intel XPU, Apple MPS
and the CPU through the same code path, because PyTorch abstracts the device for
us; the NPU backends cannot reuse it because they consume a different artefact
format entirely.

Three decisions here deserve their reasons:

* **One weight set, two wrappers.** ``AutoPipelineForImage2Image.from_pipe(t2i)``
  hands the *same* ``nn.Module`` objects to a new pipeline wrapper - it is a
  wrapper swap, not a weight copy, so img2img costs zero extra VRAM. The
  consequence is that hooks and LoRA fusions are shared: the memory plan is
  applied once, on the first wrapper, before the second exists.

* **The memory plan is an enum, not flags.** ``enable_model_cpu_offload()``
  internally calls ``.to("cpu")``, so a later ``.to("cuda")`` silently undoes it;
  ``enable_sequential_cpu_offload()`` plus ``.to()`` is a hard error in either
  order. Booleans would let a caller express those combinations.

* **Generators are always on the CPU, always rebuilt per call.** CUDA's RNG
  stream is not stable across GPU models and driver versions, while CPU RNG is
  bit-identical everywhere, so a seed keeps its meaning. A cached
  ``torch.Generator(device="cuda")`` additionally raises outright when replayed
  on a CPU or NPU backend.

Every API call below was verified against diffusers 0.40.0 as installed, not
against documentation - notably ``dtype=`` (``torch_dtype`` is deprecated and
passing both raises) and ``pipe.vae.enable_tiling()`` (the pipeline-level
``enable_vae_tiling`` was removed).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import (
    BackendUnavailableError,
    GatedModelError,
    GenerationCancelled,
    ModelLoadError,
    OutOfMemoryError,
)
from ..hardware.device import Device, DeviceKind, Runtime
from ..models.registry import EffortProfile, Img2ImgMode, LoraSpec, ModelSpec
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
from .base import Backend, BackendAvailability, BackendCapabilities, LoadOptions
from .memory import MemoryDecision, MemoryPlan, free_vram_gb, plan_memory

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

logger = logging.getLogger(__name__)


#: Neutral scheduler name -> diffusers class + extra config. ``from_config`` is
#: always used so the checkpoint's own beta schedule and ``prediction_type``
#: survive: building a scheduler from defaults silently ruins v-prediction
#: models.
_SCHEDULER_MAP: dict[SchedulerKind, tuple[str, dict[str, Any]]] = {
    SchedulerKind.DDIM: ("DDIMScheduler", {}),
    SchedulerKind.EULER: ("EulerDiscreteScheduler", {}),
    SchedulerKind.EULER_ANCESTRAL: ("EulerAncestralDiscreteScheduler", {}),
    SchedulerKind.DPMPP_2M: (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++", "solver_order": 2},
    ),
    SchedulerKind.DPMPP_2M_KARRAS: (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_karras_sigmas": True},
    ),
    SchedulerKind.LCM: ("LCMScheduler", {}),
    SchedulerKind.LMS: ("LMSDiscreteScheduler", {}),
    SchedulerKind.PNDM: ("PNDMScheduler", {}),
    SchedulerKind.FLOW_MATCH_EULER: ("FlowMatchEulerDiscreteScheduler", {}),
    SchedulerKind.UNIPC: ("UniPCMultistepScheduler", {}),
}

#: Only the StableDiffusionPipeline family accepts these. SDXL pipelines have no
#: ``safety_checker`` parameter at all, and ``from_pretrained`` treats unknown
#: component names as component overrides rather than ignoring them.
_SAFETY_CHECKER_FAMILIES = frozenset({"sd15", "sd21"})


class TorchDiffusersBackend(Backend):
    """Runs diffusers pipelines on any device PyTorch supports."""

    name: ClassVar[str] = "torch"
    runtimes: ClassVar[tuple[Runtime, ...]] = (
        Runtime.TORCH_CUDA,
        Runtime.TORCH_XPU,
        Runtime.TORCH_MPS,
        Runtime.TORCH_CPU,
    )
    priority: ClassVar[int] = 100
    model_format: ClassVar[str] = "diffusers"
    description: ClassVar[str] = "PyTorch + diffusers (CUDA, XPU, MPS, CPU)"

    def __init__(
        self,
        model: ModelSpec,
        pipeline_spec: PipelineSpec,
        options: LoadOptions | None = None,
    ) -> None:
        super().__init__(model, pipeline_spec, options)
        self._t2i: Any = None
        self._i2i: Any = None
        self._memory: MemoryDecision | None = None
        self._default_scheduler_config: dict[str, Any] | None = None
        self._applied_scheduler: SchedulerKind | None = None
        self._dropped_scheduler_config: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #

    @classmethod
    def availability(cls) -> BackendAvailability:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            return BackendAvailability(
                False,
                f"PyTorch n'est pas installe ({exc})",
                install_hint="uv pip install --index-url https://download.pytorch.org/whl/cu128 torch",
            )
        try:
            import diffusers  # noqa: F401
        except ImportError as exc:
            return BackendAvailability(
                False,
                f"diffusers n'est pas installe ({exc})",
                install_hint="uv pip install diffusers transformers accelerate safetensors",
            )
        return BackendAvailability(True)

    @staticmethod
    def check_cuda_kernels(device: Device) -> str | None:
        """Verify the installed wheel actually has kernels for this GPU.

        ``torch.cuda.is_available()`` returns ``True`` on a wheel compiled
        without the card's architecture; the failure then surfaces as "no kernel
        image is available for execution" forty seconds into a denoising loop.
        Comparing the compute capability against ``get_arch_list()`` catches it
        at startup instead. Returns an error message, or ``None`` when fine.
        """
        if device.runtime is not Runtime.TORCH_CUDA or device.compute_capability is None:
            return None
        try:
            import torch

            major, minor = device.compute_capability
            arch = f"sm_{major}{minor}"
            arch_list = torch.cuda.get_arch_list()
            if arch_list and arch not in arch_list:
                return (
                    f"La version de PyTorch installee ({torch.__version__}) ne contient pas de "
                    f"noyaux pour {device.name} ({arch}). Architectures compilees : "
                    f"{', '.join(arch_list)}."
                )
        except Exception:  # pragma: no cover - defensive
            return None
        return None

    # ------------------------------------------------------------------ #
    # Capabilities
    # ------------------------------------------------------------------ #

    @property
    def capabilities(self) -> BackendCapabilities:
        model = self.model
        components = {
            Component.TOKENIZER,
            Component.TEXT_ENCODER,
            Component.DENOISER,
            Component.VAE_DECODER,
            Component.SCHEDULER,
        }
        # A diffusers checkpoint always carries a full VAE, so the encoder is
        # there whenever the family exposes an img2img path at all.
        if model.img2img_mode is not Img2ImgMode.UNSUPPORTED:
            components.add(Component.VAE_ENCODER)
        if model.family in {"sdxl", "flux", "flux2", "chroma", "zimage"}:
            components.add(Component.TEXT_ENCODER_2)

        device = self.device
        precisions = [Precision.FP32]
        if device is not None:
            if device.supports_fp16:
                precisions.append(Precision.FP16)
            if device.supports_bf16:
                precisions.append(Precision.BF16)

        # A required scheduler is a model constraint, not a user choice: offering
        # alternatives for SDXL-Lightning or LCM would only let the user break
        # it. But the constraint belongs to the *adapter*, so an effort rung that
        # drops the adapter also lifts it - at that point the free SDXL scheduler
        # zoo is available again.
        profile = self.effort_profile
        required = None if profile.drop_lora else model.scheduler
        required = profile.scheduler if profile.scheduler is not None else required
        if required is not None and required.required:
            schedulers: tuple[SchedulerKind, ...] = (SchedulerKind.AUTO, required.kind)
        else:
            schedulers = (SchedulerKind.AUTO, *_SCHEDULER_MAP)

        return BackendCapabilities(
            components=frozenset(components),
            supports_negative_prompt=True,
            supports_guidance=True,
            supports_seed=True,
            deterministic_seed=True,
            supports_batch=True,
            supports_progress_callback=True,
            supports_cancel=True,
            dynamic_resolution=True,
            resolution_multiple=8,
            recompiles_on=frozenset(),  # PyTorch reshapes freely
            supports_compiled_cache=False,
            supported_schedulers=schedulers,
            supported_precisions=tuple(precisions),
            supports_lora=True,
            supports_controlnet=True,
            supports_ip_adapter=True,
            supports_inpainting=True,
        )

    @property
    def memory_decision(self) -> MemoryDecision | None:
        return self._memory

    def can_afford_upscale(self, scale: float) -> tuple[bool, str]:
        device = self.device
        if device is None or scale <= 1.0:
            return True, ""
        decision = plan_memory(
            self.model,
            device,
            self.pipeline_spec,
            force_offload=self.options.cpu_offload,
            force_sequential=self.options.sequential_offload,
            free_vram_gb=free_vram_gb(device),
            peak_resolution_scale=scale,
        )
        available = decision.available_gb
        if available is None:
            return True, ""
        # Sequential offload here would be slower than simply not upscaling.
        if decision.plan is MemoryPlan.SEQUENTIAL_OFFLOAD:
            return False, (
                f"un affinage a {scale:g}x demanderait l'offload sequentiel "
                f"({decision.estimated_peak_gb:.1f} Go estimes sur {available:.1f} Go libres)"
            )
        if decision.estimated_peak_gb > available:
            return False, (
                f"un affinage a {scale:g}x depasserait la VRAM disponible "
                f"({decision.estimated_peak_gb:.1f} Go estimes sur {available:.1f} Go libres)"
            )
        return True, ""

    def inspect_request(self, request: GenerationRequest) -> list[str]:
        """Warnings that need the *loaded* model to detect.

        Two silent losses live here, and both are ordinary user mistakes rather
        than exotic edge cases:

        * **CLIP truncates at 77 tokens.** Everything past that is dropped, and
          the style tags people append are exactly what gets lost.
        * **A negative prompt does nothing without guidance.** diffusers only
          runs the unconditional branch when ``guidance_scale > 1`` *and* the
          UNet has no ``time_cond_proj_dim``. On a distilled preset both fail, so
          ``--negative`` is discarded with no message at all.
        """
        warnings: list[str] = []
        pipe = self._t2i
        if pipe is None:
            return warnings

        guidance = (
            request.guidance_scale
            if request.guidance_scale is not None
            else self.model.default_guidance
        )
        denoiser = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
        time_cond = getattr(getattr(denoiser, "config", None), "time_cond_proj_dim", None)
        cfg_runs = guidance > 1.0 and time_cond is None
        if request.negative_prompt and not cfg_runs:
            reason = (
                f"guidance={guidance:g} (il faut > 1)"
                if time_cond is None
                else "le modele est distille : la branche inconditionnelle n'existe pas"
            )
            warnings.append(
                f"Le prompt negatif est ignore : {reason}. "
                "Utilisez un modele non distille, par exemple --effort high."
            )

        for text, label in ((request.prompt, "prompt"), (request.negative_prompt, "prompt negatif")):
            if not text:
                continue
            dropped = self._truncated_tokens(pipe, text)
            if dropped:
                warnings.append(
                    f"{label.capitalize()} tronque : CLIP ne lit que 77 jetons, "
                    f"{dropped} ont ete ignores (la fin du texte est perdue)."
                )
        return warnings

    @staticmethod
    def _truncated_tokens(pipe: Any, text: str) -> int:
        """How many tokens CLIP will drop from ``text``, or 0."""
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            return 0
        try:
            limit = int(getattr(tokenizer, "model_max_length", 77))
            full = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
            return max(0, len(full) - limit)
        except Exception:  # pragma: no cover - tokenizer variations
            return 0

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self, progress: ProgressCallback | None = None) -> None:
        import torch

        device = self.device
        if device is None:  # pragma: no cover - the generator always sets a plan
            raise BackendUnavailableError("Aucun peripherique selectionne.")

        kernel_error = self.check_cuda_kernels(device)
        if kernel_error:
            raise BackendUnavailableError(
                kernel_error,
                hint="Reinstallez PyTorch depuis l'index correspondant : "
                "uv pip install --index-url https://download.pytorch.org/whl/cu128 torch",
            )

        _configure_allocator(torch)
        _silence_empty_float32_notice()

        dtype = self._resolve_dtype(torch, device)
        self._memory = plan_memory(
            self.model,
            device,
            self.pipeline_spec,
            force_offload=self.options.cpu_offload,
            force_sequential=self.options.sequential_offload,
            free_vram_gb=free_vram_gb(device),
            peak_resolution_scale=self.options.peak_resolution_scale,
        )

        pipe = self._from_pretrained(torch, dtype, progress)
        self._default_scheduler_config = dict(pipe.scheduler.config)

        self._apply_lora(pipe)
        self._apply_scheduler(pipe)
        # Applied exactly once, before from_pipe creates the second wrapper:
        # the hooks live on the shared modules.
        self._apply_memory_plan(pipe, torch, device)

        pipe.set_progress_bar_config(disable=True)
        self._t2i = pipe
        self._loaded = True

    def _resolve_dtype(self, torch: Any, device: Device) -> Any:
        """Pick the load dtype, honouring a model that mandates one."""
        if self.model.force_dtype:
            # Z-Image ships fp32 shards: omitting this exhausts system RAM during
            # load, before VRAM is ever touched.
            return getattr(torch, self.model.force_dtype)
        precision = self.pipeline_spec.precision
        if precision is Precision.FP16:
            return torch.float16
        if precision is Precision.BF16:
            return torch.bfloat16
        if precision is Precision.FP32:
            return torch.float32
        # AUTO
        if device.kind is DeviceKind.CPU:
            return torch.float32  # fp16 on CPU is emulated and slower than fp32
        if device.supports_fp16:
            return torch.float16
        return torch.float32

    def _build_load_kwargs(self, torch: Any, dtype: Any) -> dict[str, Any]:
        model = self.model
        kwargs: dict[str, Any] = {"dtype": dtype, "use_safetensors": True}
        if model.family == "sdxl":
            # Only the SDXL pipelines take this. diffusers enables the invisible
            # watermarker from mere package presence: nothing installs it today,
            # but a future transitive dependency would otherwise start altering
            # every image without a word.
            kwargs["add_watermarker"] = False
        if model.has_fp16_variant and dtype is torch.float16:
            kwargs["variant"] = "fp16"
        if model.revision:
            kwargs["revision"] = model.revision
        if self.options.cache_dir:
            kwargs["cache_dir"] = self.options.cache_dir
        if self.options.local_files_only:
            kwargs["local_files_only"] = True
        if self.options.token:
            kwargs["token"] = self.options.token
        if model.family in _SAFETY_CHECKER_FAMILIES and not self.options.safety_checker:
            # Both are needed: safety_checker=None disables the check, and
            # requires_safety_checker=False silences the warning about it.
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False
        return kwargs

    def _from_pretrained(self, torch: Any, dtype: Any, progress: ProgressCallback | None) -> Any:
        import diffusers

        model = self.model
        kwargs = self._build_load_kwargs(torch, dtype)

        if model.vae_repo:
            # The stock SDXL fp16 VAE overflows to NaN and yields black images.
            kwargs["vae"] = diffusers.AutoencoderKL.from_pretrained(
                model.vae_repo,
                dtype=dtype,
                cache_dir=self.options.cache_dir,
                local_files_only=self.options.local_files_only,
            )

        pipeline_cls = getattr(diffusers, model.txt2img_class, None)
        if pipeline_cls is None:
            # A newer preset on an older diffusers: fall back to the auto class,
            # which resolves from the repo's model_index.json.
            logger.debug("%s absent de diffusers, repli sur AutoPipeline", model.txt2img_class)
            pipeline_cls = diffusers.AutoPipelineForText2Image

        try:
            return pipeline_cls.from_pretrained(model.repo_id, **kwargs)
        except Exception as exc:
            raise self._translate_load_error(exc, kwargs) from exc

    def _translate_load_error(self, exc: Exception, kwargs: dict[str, Any]) -> Exception:
        """Turn library exceptions into errors that say what to do next."""
        text = str(exc)
        model = self.model
        if "401" in text or "gated" in text.lower() or "GatedRepo" in type(exc).__name__:
            return GatedModelError(
                f"Le depot '{model.repo_id}' est sous accord d'utilisation.",
                hint=f"Executez 'hf auth login', puis acceptez la licence sur "
                f"https://huggingface.co/{model.repo_id}",
            )
        if "variant" in text and "fp16" in text:
            return ModelLoadError(
                f"Le depot '{model.repo_id}' ne publie pas de variante fp16.",
                hint="Signalez-le dans le registre avec has_fp16_variant=False.",
            )
        if isinstance(exc, OSError) and (
            "Connection" in text or "offline" in text.lower() or "local_files_only" in text
        ):
            return ModelLoadError(
                f"Impossible de recuperer '{model.repo_id}'.",
                hint="Verifiez la connexion reseau, ou pre-telechargez le modele avec "
                f"'imagegen download {model.key}'.",
            )
        return ModelLoadError(f"Chargement de '{model.repo_id}' impossible : {exc}")

    @property
    def effort_profile(self) -> EffortProfile:
        """The model's settings at this pipeline's effort level."""
        return self.model.effort(self.pipeline_spec.effort)

    def _resolved_lora(self) -> LoraSpec | None:
        """The adapter to fuse, after the effort level has had its say.

        An effort level can point at a different adapter (SDXL-Lightning ships
        2-, 4- and 8-step LoRAs over the same base weights) or drop it entirely,
        which is what the top of the ladder does to run the base model with real
        classifier-free guidance.
        """
        profile = self.effort_profile
        if profile.drop_lora:
            return None
        return profile.lora or self.model.lora

    def _apply_lora(self, pipe: Any) -> None:
        """Fuse the resolved adapter, if any.

        Fusing happens at load and never per call: ``from_pipe`` shares the
        denoiser between the txt2img and img2img wrappers, so a later fuse would
        mutate both.
        """
        lora = self._resolved_lora()
        if lora is None:
            return
        try:
            load_kwargs: dict[str, Any] = {}
            if lora.weight_name:
                load_kwargs["weight_name"] = lora.weight_name
            if self.options.cache_dir:
                load_kwargs["cache_dir"] = self.options.cache_dir
            pipe.load_lora_weights(lora.repo_id, **load_kwargs)
            if lora.fuse:
                pipe.fuse_lora(lora_scale=lora.scale)
        except Exception as exc:
            raise ModelLoadError(
                f"Chargement de l'adaptateur LoRA '{lora.repo_id}' impossible : {exc}",
                hint="Verifiez que 'peft' est installe : uv pip install peft",
            ) from exc

    def _apply_scheduler(self, pipe: Any) -> None:
        """Swap the scheduler, keeping the checkpoint's own config as the base."""
        import diffusers

        # An effort level that drops the adapter also drops the adapter's
        # mandatory scheduler: Euler-trailing at guidance 0 is a Lightning
        # requirement, not an SDXL one.
        profile = self.effort_profile
        spec = profile.scheduler if profile.scheduler is not None else (
            None if profile.drop_lora else self.model.scheduler
        )
        requested = self.pipeline_spec.scheduler
        if requested is SchedulerKind.AUTO:
            if spec is None:
                return  # keep the checkpoint's own scheduler
            kind, extra = spec.kind, dict(spec.config)
        else:
            if spec is not None and spec.required and requested is not spec.kind:
                logger.warning(
                    "Le modele %s impose le scheduler %s : '%s' est ignore.",
                    self.model.key,
                    spec.kind.value,
                    requested.value,
                )
                kind, extra = spec.kind, dict(spec.config)
            else:
                kind = requested
                extra = dict(spec.config) if spec is not None and spec.kind is requested else {}

        mapping = _SCHEDULER_MAP.get(kind)
        if mapping is None:
            return
        class_name, base_config = mapping
        scheduler_cls = getattr(diffusers, class_name, None)
        if scheduler_cls is None:
            logger.warning("Scheduler %s absent de diffusers, conservation du defaut.", class_name)
            return

        config = {**base_config, **extra}
        try:
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **config)
            self._applied_scheduler = kind
        except Exception as exc:
            logger.warning("Scheduler %s inapplicable (%s), conservation du defaut.", class_name, exc)

    def _apply_memory_plan(self, pipe: Any, torch: Any, device: Device) -> None:
        """Apply exactly one placement strategy, plus the orthogonal VAE knobs."""
        decision = self._memory
        assert decision is not None

        # 1. VAE knobs first: plain attribute flags, no hooks, order-independent.
        if decision.vae_slicing:
            _try_vae(pipe, "enable_slicing")
        if decision.vae_tiling:
            _try_vae(pipe, "enable_tiling")

        # 2. Exactly one placement strategy.
        target = "cpu" if decision.plan is MemoryPlan.CPU else device.handle
        if decision.plan is MemoryPlan.FULL_GPU:
            pipe.to(target)
        elif decision.plan is MemoryPlan.CPU:
            pipe.to("cpu")
        elif decision.plan is MemoryPlan.MODEL_OFFLOAD:
            # Never followed by .to(device): the call itself moves the pipeline
            # to the CPU and installs hooks that a later .to() would defeat.
            pipe.enable_model_cpu_offload(device=target)
        elif decision.plan is MemoryPlan.SEQUENTIAL_OFFLOAD:
            pipe.enable_sequential_cpu_offload(device=target)

        # 3. Compilation last, and only on request: Windows has no default Triton
        # wheel, so inductor usually falls back to eager, and torch.compile
        # recompiles on every resolution change - poor for a CLI.
        if self.options.compile_model:
            denoiser = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
            if denoiser is not None:
                try:
                    compiled = torch.compile(denoiser, mode="reduce-overhead", fullgraph=True)
                    if hasattr(pipe, "unet"):
                        pipe.unet = compiled
                    else:
                        pipe.transformer = compiled
                except Exception as exc:
                    logger.warning("torch.compile indisponible (%s), execution en mode eager.", exc)

    def _image_pipeline(self) -> Any:
        """The img2img wrapper, built on first use over the same weights."""
        if self._i2i is not None:
            return self._i2i
        import diffusers

        model = self.model
        if model.img2img_mode is Img2ImgMode.SAME_CLASS_IMAGE_ARG:
            # FLUX.2-klein, FLUX.1-Kontext: one class does both modes.
            self._i2i = self._t2i
            return self._i2i

        target_cls = getattr(diffusers, model.img2img_class or "", None)
        try:
            if target_cls is not None:
                self._i2i = target_cls.from_pipe(self._t2i)
            else:
                self._i2i = diffusers.AutoPipelineForImage2Image.from_pipe(self._t2i)
        except Exception as exc:
            raise ModelLoadError(
                f"Impossible de construire le pipeline image-to-image pour '{model.key}' : {exc}"
            ) from exc
        # set_progress_bar_config is a per-instance attribute: from_pipe does not
        # carry it over, so the second wrapper needs its own call.
        self._i2i.set_progress_bar_config(disable=True)
        return self._i2i

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> list[GeneratedImage]:
        import torch

        if not self._loaded:
            raise ModelLoadError("Le backend n'est pas charge : appelez load() d'abord.")

        pipe = self._image_pipeline() if request.mode is Mode.IMAGE_TO_IMAGE else self._t2i
        steps = request.steps or self.model.default_steps
        guidance = (
            request.guidance_scale
            if request.guidance_scale is not None
            else self.model.default_guidance
        )

        generators, seeds = _make_generators(torch, request.seed, request.num_images)
        total_steps = (
            min(int(steps * request.strength), steps)
            if request.mode is Mode.IMAGE_TO_IMAGE
            else steps
        )
        callback = _make_step_callback(max(total_steps, 1), progress, cancel)

        call_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_images_per_prompt": request.num_images,
            "generator": generators,
            "output_type": "pil",
            "callback_on_step_end": callback,
            # Nothing is read back from the callback, so ask for no tensors.
            "callback_on_step_end_tensor_inputs": [],
        }
        if request.negative_prompt:
            call_kwargs["negative_prompt"] = request.negative_prompt

        if request.mode is Mode.IMAGE_TO_IMAGE:
            # img2img has no width/height parameters: output size comes from the
            # source image, which the generator has already resized.
            call_kwargs["image"] = request.init_image
            call_kwargs["strength"] = request.strength
        else:
            call_kwargs["width"] = self.pipeline_spec.width
            call_kwargs["height"] = self.pipeline_spec.height

        # Model-mandated extras, then the effort rung's own, then the caller's:
        # each layer is more specific than the one before.
        call_kwargs.update(self.model.call_kwargs)
        call_kwargs.update(self.effort_profile.call_kwargs)
        call_kwargs.update(request.extra)

        try:
            output = pipe(**call_kwargs)
        except GenerationCancelled:
            raise
        except torch.cuda.OutOfMemoryError as exc:
            raise self._translate_oom(exc) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise self._translate_oom(exc) from exc
            raise
        finally:
            # Interrupting mid-loop under offload can leave hooks installed.
            maybe_free = getattr(pipe, "maybe_free_model_hooks", None)
            if maybe_free is not None:
                with contextlib.suppress(Exception):  # best effort cleanup
                    maybe_free()

        images: list[Image] = list(output.images)
        parameters = self._record_parameters(request, steps, guidance)
        return [
            GeneratedImage(
                image=image,
                seed=seeds[index] if index < len(seeds) else seeds[-1],
                index=index,
                parameters={**parameters, "seed": seeds[index] if index < len(seeds) else seeds[-1]},
            )
            for index, image in enumerate(images)
        ]

    def _translate_oom(self, exc: Exception) -> OutOfMemoryError:
        decision = self._memory
        plan = decision.plan.value if decision else "?"
        hints = []
        if decision and decision.plan is MemoryPlan.FULL_GPU:
            hints.append("relancez avec --cpu-offload")
        elif decision and decision.plan is MemoryPlan.MODEL_OFFLOAD:
            hints.append("relancez avec --sequential-offload")
        spec = self.pipeline_spec
        if spec.width > 768 or spec.height > 768:
            hints.append(f"reduisez la resolution (actuellement {spec.width}x{spec.height})")
        if spec.batch_size > 1:
            hints.append("generez une image a la fois (-n 1)")
        hints.append("ou choisissez un modele plus leger (imagegen models)")
        return OutOfMemoryError(
            f"Memoire GPU insuffisante avec le plan '{plan}'.",
            hint="Essayez : " + ", ".join(hints) + ".",
        )

    def _record_parameters(
        self, request: GenerationRequest, steps: int, guidance: float
    ) -> dict[str, Any]:
        device = self.device
        scheduler = self._applied_scheduler.value if self._applied_scheduler else "defaut"
        return {
            "model": self.model.key,
            "repo_id": self.model.repo_id,
            "backend": self.name,
            "device": device.short_id if device else "?",
            "mode": request.mode.value,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": steps,
            "effective_steps": (
                min(int(steps * request.strength), steps)
                if request.mode is Mode.IMAGE_TO_IMAGE
                else steps
            ),
            "guidance_scale": guidance,
            "strength": request.strength if request.mode is Mode.IMAGE_TO_IMAGE else None,
            "width": self.pipeline_spec.width,
            "height": self.pipeline_spec.height,
            "scheduler": scheduler,
            "precision": self.pipeline_spec.precision.value,
            "memory_plan": self._memory.plan.value if self._memory else None,
        }

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #

    def memory_usage_bytes(self) -> int | None:
        device = self.device
        if device is None or device.runtime is not Runtime.TORCH_CUDA:
            return None
        try:
            import torch

            return int(torch.cuda.memory_allocated(device.index))
        except Exception:
            return None

    def unload(self) -> None:
        """Release the weights *and* the offload hooks.

        Dropping the references is not enough. ``enable_model_cpu_offload``
        installs accelerate hooks that hold their own references to every
        sub-model, so without ``remove_all_hooks`` the previous pipeline stays
        resident. Measured: loading a second pipeline in the same process after
        a plain ``del`` left 7.9 GB of 8 GB occupied and the next generation took
        38 s instead of 7 s - the WDDM driver spilling to system RAM rather than
        raising. Switching model or effort level goes through exactly this path.
        """
        for pipe in (self._i2i, self._t2i):
            if pipe is None:
                continue
            for method in ("remove_all_hooks", "maybe_free_model_hooks"):
                function = getattr(pipe, method, None)
                if function is not None:
                    with contextlib.suppress(Exception):
                        function()
        self._i2i = None
        self._t2i = None
        self._loaded = False
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:  # pragma: no cover - best effort
            pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _EmptyFloat32Filter(logging.Filter):
    """Drop one false-positive diffusers notice, and only when it is false.

    ``from_pipe`` calls ``to(dtype=...)`` on the shared modules, so diffusers
    logs that some modules "should be kept in float32" - followed by an *empty*
    list. That fires on every img2img run and means nothing. The match requires
    the empty brackets, so a genuine list still reaches the user.
    """

    _NOISE = "should be kept in float32: []"

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return self._NOISE not in record.getMessage()
        except Exception:  # pragma: no cover - a malformed record is not ours to drop
            return True


_FLOAT32_FILTER = _EmptyFloat32Filter()


def _silence_empty_float32_notice() -> None:
    """Attach the filter where diffusers actually emits.

    It is a ``logger.warning``, not a ``warnings.warn``, and it comes from a
    child logger. diffusers keeps its handler on the ``diffusers`` root with
    ``propagate=False``, so the filter has to sit on that *handler* - a filter on
    the logger itself would never see records propagated up from children.
    """
    logger_obj = logging.getLogger("diffusers")
    targets = list(logger_obj.handlers)
    if not targets:  # pragma: no cover - handler is installed on first use
        logger_obj.addFilter(_FLOAT32_FILTER)
        return
    for handler in targets:
        if _FLOAT32_FILTER not in handler.filters:
            handler.addFilter(_FLOAT32_FILTER)


def _configure_allocator(torch: Any) -> None:
    """Ask the CUDA allocator for expandable segments.

    A long-lived CLI that generates repeatedly fragments an 8 GB pool until an
    allocation that would have fit no longer does. The variable was renamed in
    torch 2.9 - setting the old name there emits a deprecation warning on every
    run, so pick by version.
    """
    try:
        version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:  # pragma: no cover - unusual version strings
        version = (0, 0)
    name = "PYTORCH_ALLOC_CONF" if version >= (2, 9) else "PYTORCH_CUDA_ALLOC_CONF"
    os.environ.setdefault(name, "expandable_segments:True")


def _try_vae(pipe: Any, method: str) -> None:
    """Call a VAE memory knob when the VAE actually implements it.

    ``AutoencoderMixin`` raises ``NotImplementedError`` for VAEs without the
    feature (``AutoencoderKLWan``, ``AsymmetricAutoencoderKL``), and older
    diffusers exposed these on the pipeline instead.
    """
    vae = getattr(pipe, "vae", None)
    function = getattr(vae, method, None) if vae is not None else None
    if function is None:
        return
    try:
        function()
    except NotImplementedError:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("vae.%s a echoue : %s", method, exc)


def _make_generators(torch: Any, seed: int | None, count: int) -> tuple[list[Any], list[int]]:
    """One CPU generator per image, with explicit per-image seeds.

    CPU generators are used unconditionally: CUDA's RNG stream is not guaranteed
    stable across GPU models and driver versions, so a seed would not keep its
    meaning, and a ``device="cuda"`` generator raises outright if the same seed is
    later replayed on a CPU or NPU backend. diffusers slices the noise per batch
    element, so seed ``seeds[i]`` alone reproduces image ``i`` exactly.
    """
    if seed is None:
        seed = int(torch.seed()) & 0xFFFF_FFFF
    seeds = [(seed + index) & 0xFFFF_FFFF_FFFF_FFFF for index in range(max(count, 1))]
    generators = [torch.Generator(device="cpu").manual_seed(value) for value in seeds]
    return generators, seeds


def _make_step_callback(
    total_steps: int,
    progress: ProgressCallback | None,
    cancel: CancelCallback | None,
) -> Any:
    """Build a ``callback_on_step_end`` that reports progress and honours cancel.

    diffusers offers no abort return value, so cancellation raises out of the
    denoising loop; nothing inside catches it.
    """

    def _callback(pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict) -> dict:
        if cancel is not None and cancel():
            raise GenerationCancelled()
        if progress is not None:
            progress(min(step_index + 1, total_steps), total_steps)
        return {}  # must be a dict: the pipeline calls .pop on it

    return _callback
