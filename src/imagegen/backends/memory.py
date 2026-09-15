"""Memory planning for the PyTorch backend.

An 8 GB card is not "a GPU with enough memory": SDXL in fp16 is ~7 GB of weights
before a single activation is allocated, and the desktop compositor has already
taken a few hundred megabytes. Deciding placement by trial-and-error means the
user meets a CUDA OOM traceback; deciding it up front means they get a sentence
explaining what will happen and why.

On Windows the stakes are higher than "an exception". Measured on this project's
reference machine (RTX 5050, 8 GB), SDXL at 1024px forced fully resident did not
raise ``OutOfMemoryError`` at all: the WDDM driver spilled to system RAM and the
generation took **190 s instead of 13 s** with model offload. Exceeding VRAM on
Windows is usually a silent 15x slowdown, not a crash - which is why the estimate
must be right *before* loading, and why an over-capacity plan emits a warning
rather than relying on an OOM that may never come.

The placement strategies are **mutually exclusive by construction**. diffusers
enforces this at runtime - ``enable_sequential_cpu_offload()`` followed by
``.to("cuda")`` raises, ``enable_model_cpu_offload()`` internally calls
``.to("cpu")`` so a later ``.to("cuda")`` silently undoes it - so they are
modelled as one :class:`MemoryPlan` enum value rather than a set of booleans.
Booleans would make the illegal combinations representable.

The VAE knobs are orthogonal and stack freely on top of any plan.

This module imports torch only inside functions: it is read by ``imagegen
doctor`` on machines where torch may not be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..hardware.device import Device, DeviceKind
from ..models.registry import ModelSpec, Tier
from ..types import PipelineSpec


class MemoryPlan(str, Enum):
    """Where the weights live during inference. Exactly one applies."""

    #: ``pipe.to(device)``. Fastest. Everything resident.
    FULL_GPU = "full-gpu"
    #: ``pipe.enable_model_cpu_offload()``. One whole sub-model on the GPU at a
    #: time. A few percent slower on SD1.5, more on SDXL (two text encoders).
    MODEL_OFFLOAD = "model-offload"
    #: ``pipe.enable_sequential_cpu_offload()``. Submodule granularity. The
    #: diffusers docs call it "extremely slow"; last resort only.
    SEQUENTIAL_OFFLOAD = "sequential-offload"
    #: ``pipe.to("cpu")``. Minutes per image, but it always works.
    CPU = "cpu"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """The chosen plan plus the reasoning, so the CLI can explain itself."""

    plan: MemoryPlan
    vae_slicing: bool
    vae_tiling: bool
    reason: str
    estimated_peak_gb: float
    available_gb: float | None
    warnings: tuple[str, ...] = ()

    def describe(self) -> str:
        where = {
            MemoryPlan.FULL_GPU: "poids entierement en VRAM",
            MemoryPlan.MODEL_OFFLOAD: "offload par sous-modele (CPU <-> GPU)",
            MemoryPlan.SEQUENTIAL_OFFLOAD: "offload sequentiel (tres lent)",
            MemoryPlan.CPU: "calcul sur CPU",
        }[self.plan]
        return f"{where} - {self.reason}"


#: Kept free for the driver, the desktop compositor and allocator fragmentation.
_HEADROOM_GB = 0.6

#: Share of the weights held by the denoiser (UNet or DiT), which is the single
#: largest component and therefore what sets the peak under model offload.
#: SDXL is 5.14 GB of UNet out of 6.95 GB total; SD 1.5 is 1.7 of 2.6.
_DENOISER_SHARE = 0.72


def estimate_activation_gb(
    spec: PipelineSpec, model: ModelSpec, peak_scale: float = 1.0
) -> float:
    """Peak activation memory, in GB, beyond the weights.

    Fitted to measurements on this project's reference card (RTX 5050, 8 GB,
    SDXL under model offload, VAE tiling and slicing on), because the first
    version of this function overestimated by roughly 3x and being pessimistic
    here is not "safe": it downgrades a working plan to a much slower one.

    Measured peak *total* allocation, from which the activation share is the
    remainder above the 5.14 GB SDXL UNet:

        1024px, batch 1, no CFG   5.43 GB  ->  ~0.29 GB
        1024px, batch 2 (CFG 7)   5.59 GB  ->  ~0.45 GB
         768px, batch 2 (CFG 7)   5.45 GB  ->  ~0.31 GB

    SDPA is memory-efficient by default on torch >= 2.0, which is why the
    numbers are this flat. The constants below sit slightly above the fit.
    """
    # A refinement pass denoises an upscaled copy, so the peak is set by that
    # larger canvas rather than by the resolution in the spec.
    megapixels = (spec.width * spec.height * peak_scale**2) / (1024 * 1024)
    batch = max(spec.batch_size, 1)
    # CFG runs the denoiser on a doubled batch (unconditional + conditional).
    if spec.guidance_enabled:
        batch *= 2
    return 0.25 + 0.18 * megapixels * batch


def plan_memory(
    model: ModelSpec,
    device: Device,
    spec: PipelineSpec,
    *,
    force_offload: bool | None = None,
    force_sequential: bool = False,
    free_vram_gb: float | None = None,
    peak_resolution_scale: float = 1.0,
) -> MemoryDecision:
    """Choose a placement strategy for ``model`` on ``device``.

    ``force_offload`` / ``force_sequential`` come from the CLI and win over the
    automatic decision, which is what makes the automation debuggable.
    """
    warnings: list[str] = []
    activation_gb = estimate_activation_gb(spec, model, peak_resolution_scale)
    weights_gb = model.weights_vram_gb or 0.0
    peak_gb = weights_gb + activation_gb

    # VAE decode is the usual OOM site at 1024px: tiling caps it, and it is a
    # documented no-op below the VAE's internal threshold, so it is free to
    # leave on. Slicing only matters for batches, and costs nothing otherwise.
    peak_side = max(spec.width, spec.height) * peak_resolution_scale
    vae_tiling = peak_side >= 1024
    vae_slicing = spec.batch_size > 1

    if device.kind is DeviceKind.CPU:
        return MemoryDecision(
            MemoryPlan.CPU,
            vae_slicing,
            vae_tiling,
            "aucun accelerateur selectionne",
            peak_gb,
            device.total_memory_gb,
            ("La generation sur CPU prend plusieurs minutes par image.",),
        )

    if force_sequential:
        return MemoryDecision(
            MemoryPlan.SEQUENTIAL_OFFLOAD,
            vae_slicing,
            True,
            "force par --sequential-offload",
            peak_gb,
            free_vram_gb,
            ("L'offload sequentiel est tres lent : a reserver aux cas d'OOM persistant.",),
        )

    available = free_vram_gb if free_vram_gb is not None else device.total_memory_gb

    if force_offload is True:
        return MemoryDecision(
            MemoryPlan.MODEL_OFFLOAD, vae_slicing, True, "force par --cpu-offload", peak_gb, available
        )
    if force_offload is False:
        if available is not None and peak_gb + _HEADROOM_GB > available:
            warnings.append(
                f"--no-cpu-offload demande alors que le pic estime ({peak_gb:.1f} Go) depasse "
                f"la VRAM libre ({available:.1f} Go) : un OOM est probable."
            )
        return MemoryDecision(
            MemoryPlan.FULL_GPU,
            vae_slicing,
            vae_tiling,
            "force par --no-cpu-offload",
            peak_gb,
            available,
            tuple(warnings),
        )

    if model.tier is Tier.QUANTIZED:
        warnings.append(
            f"'{model.key}' exige une quantification pour tenir en 8 Go "
            f"({model.weights_vram_gb:.1f} Go de poids en fp16/bf16)."
        )

    if available is None:
        # Shared-memory device (NPU, iGPU) or a runtime that cannot report:
        # offloading is the safe default rather than a gamble.
        return MemoryDecision(
            MemoryPlan.MODEL_OFFLOAD,
            vae_slicing,
            vae_tiling,
            "memoire du peripherique inconnue, choix prudent",
            peak_gb,
            None,
            tuple(warnings),
        )

    if peak_gb + _HEADROOM_GB <= available:
        return MemoryDecision(
            MemoryPlan.FULL_GPU,
            vae_slicing,
            vae_tiling,
            f"pic estime {peak_gb:.1f} Go sur {available:.1f} Go libres",
            peak_gb,
            available,
            tuple(warnings),
        )

    # Under model offload exactly one sub-model sits on the GPU at a time, so
    # the peak is set by the denoiser, not by the whole checkpoint.
    denoiser_gb = weights_gb * _DENOISER_SHARE
    offload_peak = denoiser_gb + activation_gb
    if offload_peak + _HEADROOM_GB <= available:
        return MemoryDecision(
            MemoryPlan.MODEL_OFFLOAD,
            vae_slicing,
            True,
            f"poids ({weights_gb:.1f} Go) trop volumineux pour {available:.1f} Go libres, "
            f"offload par sous-modele (pic ~{offload_peak:.1f} Go)",
            offload_peak,
            available,
            tuple(warnings),
        )

    # Sequential offload is documented as "extremely slow", so it is reserved for
    # the case model offload genuinely cannot serve: the denoiser alone does not
    # fit. Between the two, VAE tiling and slicing cut the activation peak
    # enough that a marginal miss on the estimate is worth attempting.
    if denoiser_gb + _HEADROOM_GB <= available:
        warnings.append(
            f"Marge faible : le debruiteur seul occupe ~{denoiser_gb:.1f} Go sur "
            f"{available:.1f} Go libres. En cas d'echec memoire, reduisez la resolution "
            "ou ajoutez --sequential-offload."
        )
        return MemoryDecision(
            MemoryPlan.MODEL_OFFLOAD,
            vae_slicing,
            True,
            f"offload par sous-modele, marge etroite (debruiteur ~{denoiser_gb:.1f} Go)",
            offload_peak,
            available,
            tuple(warnings),
        )

    warnings.append(
        f"Le debruiteur seul ({denoiser_gb:.1f} Go) ne tient pas dans {available:.1f} Go : "
        "repli sur l'offload sequentiel, tres lent. Reduisez --width/--height ou choisissez "
        "un modele plus leger (imagegen models)."
    )
    return MemoryDecision(
        MemoryPlan.SEQUENTIAL_OFFLOAD,
        vae_slicing,
        True,
        f"debruiteur ~{denoiser_gb:.1f} Go > {available:.1f} Go libres",
        offload_peak,
        available,
        tuple(warnings),
    )


def free_vram_gb(device: Device) -> float | None:
    """Actually-free VRAM, which is what matters on a desktop.

    ``torch.cuda.mem_get_info`` reports what the driver has left after the
    compositor and every other process, so it is preferred over the card's
    nominal size.
    """
    if device.runtime.value != "torch-cuda":
        return device.total_memory_gb
    try:
        import torch

        free_bytes, _total = torch.cuda.mem_get_info(device.index)
        return free_bytes / (1024**3)
    except Exception:
        return device.total_memory_gb
