"""Terminal interface.

The commands are shaped so that a user can answer three questions without
reading any code: what can this machine run (``devices``, ``backends``,
``doctor``), what can it generate (``models``), and then generate it
(``generate``). One command covers both modes: passing ``--image`` switches to
image-to-image, because from the user's point of view it is the same request
with one more input.

Errors are printed as a message plus the concrete next action, never as a
traceback - unless ``--debug`` is passed, which is what a bug report needs.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from . import __version__
from .errors import ConfigurationError, GenerationCancelled, ImageGenError
from .types import DEFAULT_EFFORT, Effort, GenerationRequest, Precision, SchedulerKind

console = Console()
error_console = Console(stderr=True)

app = typer.Typer(
    name="imagegen",
    help="Generation d'images a partir d'un prompt et, optionnellement, d'une image source.",
    no_args_is_help=True,
    add_completion=False,
)

# Silences the Windows symlink warning from the HF cache. Duplicated files cost
# disk, not correctness, and the warning frightens users into thinking the
# download failed.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

#: Read once, at import: typer evaluates option defaults when the command is
#: declared. ``IMAGEGEN_EFFORT`` lets a user pin a level for a whole session
#: without repeating the flag; ``--effort`` still wins over it.
_DEFAULT_EFFORT_VALUE = os.environ.get("IMAGEGEN_EFFORT", DEFAULT_EFFORT.value)


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #

@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Description de l'image a produire."),
    image: Path | None = typer.Option(
        None, "-i", "--image", help="Image source : bascule en image-to-image."
    ),
    strength: float = typer.Option(
        0.65,
        "--strength",
        min=0.01,
        max=1.0,
        help="Intensite de transformation de l'image source. 0.3 reste proche, 0.9 s'en eloigne.",
    ),
    negative: str | None = typer.Option(
        None, "-N", "--negative", help="Ce que l'image ne doit pas contenir."
    ),
    model: str | None = typer.Option(
        None, "-m", "--model", help="Cle du modele (voir 'imagegen models')."
    ),
    effort: str = typer.Option(
        _DEFAULT_EFFORT_VALUE,
        "-e",
        "--effort",
        envvar="IMAGEGEN_EFFORT",
        help="Niveau d'effort : draft, fast, balanced, high, max. Regle d'un coup les etapes, "
        "l'adaptateur, l'echantillonneur et la passe d'affinage. Les options explicites "
        "(--steps, --guidance, --scheduler) restent prioritaires. Reglable durablement via "
        "la variable d'environnement IMAGEGEN_EFFORT.",
    ),
    steps: int | None = typer.Option(None, "-s", "--steps", help="Nombre d'etapes de debruitage."),
    guidance: float | None = typer.Option(
        None, "-g", "--guidance", help="Force du guidage textuel (CFG). Doit valoir 0 sur les modeles distilles."
    ),
    width: int | None = typer.Option(None, "-W", "--width", help="Largeur, multiple de 8."),
    height: int | None = typer.Option(None, "-H", "--height", help="Hauteur, multiple de 8."),
    count: int = typer.Option(1, "-n", "--count", min=1, help="Nombre d'images a generer."),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Graine. A parametres identiques (dont -n), la meme graine redonne exactement la "
        "meme image. En changeant -n ou de backend, l'image reste la meme mais pas au pixel pres.",
    ),
    scheduler: str | None = typer.Option(
        None, "--scheduler", help="Scheduler (auto, euler, euler-a, dpmpp-2m-karras, lcm...)."
    ),
    output: Path = typer.Option(Path("outputs"), "-o", "--output", help="Dossier de sortie."),
    device: str = typer.Option("auto", "--device", help="auto, cuda, cuda:0, cpu, npu, ov:NPU..."),
    backend: str | None = typer.Option(None, "--backend", help="torch, openvino, onnx."),
    precision: str = typer.Option("auto", "--precision", help="auto, fp32, fp16, bf16."),
    cpu_offload: bool | None = typer.Option(
        None, "--cpu-offload/--no-cpu-offload", help="Force l'offload CPU des sous-modeles."
    ),
    sequential_offload: bool = typer.Option(
        False, "--sequential-offload", help="Offload sequentiel : tres lent, dernier recours."
    ),
    safety_checker: bool = typer.Option(
        False, "--safety-checker/--no-safety-checker", help="Filtre de contenu (SD 1.x uniquement)."
    ),
    translate: str = typer.Option(
        "auto",
        "--translate",
        help="auto | always | never. Les encodeurs de texte de SD/SDXL sont entraines en "
        "anglais : un prompt francais perd silencieusement des mots. 'auto' traduit quand un "
        "prompt non anglais est detecte, et affiche toujours la traduction.",
    ),
    offline: bool = typer.Option(False, "--offline", help="N'effectue aucun appel reseau."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Dossier de cache des modeles."),
    sidecar: bool = typer.Option(False, "--sidecar", help="Ecrit aussi un .json de parametres."),
    as_json: bool = typer.Option(False, "--json", help="Sortie machine."),
    debug: bool = typer.Option(False, "--debug", help="Affiche les traces completes."),
) -> None:
    """Genere une ou plusieurs images."""
    _configure_logging(debug)
    from .backends import LoadOptions
    from .generation import ImageGenerator, build_output_path, save_image

    try:
        options = LoadOptions(
            cpu_offload=cpu_offload,
            sequential_offload=sequential_offload,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=offline,
            safety_checker=safety_checker,
        )
        generator = ImageGenerator.create(
            model=model,
            device=device,
            backend=backend,
            options=options,
            precision=Precision(precision),
            effort=_parse_effort(effort),
            translate=_parse_choice(translate, {"auto", "always", "never"}, "--translate"),
        )

        request = GenerationRequest(
            prompt=prompt,
            negative_prompt=negative,
            init_image=image,
            strength=strength,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance,
            scheduler=SchedulerKind(scheduler) if scheduler else None,
            seed=seed,
            num_images=count,
        )

        plan = generator.build_spec(request)
        if not as_json:
            _print_plan(generator, plan)

        cancelled = _install_interrupt_handler()

        with _progress(as_json) as (progress_bar, task_id):
            def on_step(step: int, total: int) -> None:
                if progress_bar is not None:
                    progress_bar.update(task_id, completed=step, total=total)

            result = generator.generate(
                request,
                progress=on_step,
                cancel=lambda: cancelled(),
                plan=plan,
            )
        # The result keeps every warning for --json and library consumers; the
        # terminal skips the planning ones, already shown before the work began.
        already_shown = set(plan.warnings) if not as_json else set()

        for generated in result.images:
            path = build_output_path(output, prompt, generated.seed, generated.index)
            save_image(generated.image, path, generated.parameters, sidecar=sidecar)
            result.paths.append(path)

        if as_json:
            console.print_json(
                json.dumps(
                    {
                        "images": [str(p) for p in result.paths],
                        "seeds": [g.seed for g in result.images],
                        "model": result.model_key,
                        "effort": plan.spec.effort.value,
                        "prompt": plan.request.prompt,
                        "prompt_original": prompt if plan.request.prompt != prompt else None,
                        "backend": result.backend,
                        "device": result.device,
                        "mode": result.mode.value,
                        "duration_s": round(result.duration_s, 2),
                        "load_s": round(result.load_s, 2),
                        "warnings": result.warnings,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            _print_result(result, skip=already_shown)

    except GenerationCancelled:
        error_console.print("[yellow]Generation interrompue.[/yellow]")
        raise typer.Exit(130) from None
    except ImageGenError as exc:
        _fail(exc, debug)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        error_console.print("[yellow]Interrompu.[/yellow]")
        raise typer.Exit(130) from None


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

models_app = typer.Typer(help="Catalogue des modeles.", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list(
    all_models: bool = typer.Option(
        False, "-a", "--all", help="Inclut les modeles optionnels (licences restrictives, lourds)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Sortie machine."),
) -> None:
    """Liste les modeles disponibles."""
    from .models import list_models

    specs = list_models(include_opt_in=all_models)
    if as_json:
        console.print_json(
            json.dumps(
                [
                    {
                        "key": s.key,
                        "repo_id": s.repo_id,
                        "license": s.license.name,
                        "commercial": s.license.commercial.value,
                        "gated": s.license.gated,
                        "steps": s.effort(DEFAULT_EFFORT).steps,
                        "guidance": s.default_guidance,
                        "size": f"{s.default_width}x{s.default_height}",
                        "download_gb": s.download_gb,
                        "tier": s.tier.value,
                        "img2img": s.supports_img2img,
                    }
                    for s in specs
                ],
                ensure_ascii=False,
            )
        )
        return

    table = Table(title="Modeles", show_lines=False)
    table.add_column("cle", style="bold cyan")
    table.add_column("taille", justify="right")
    table.add_column("etapes", justify="right")
    table.add_column("telech.", justify="right")
    table.add_column("VRAM", justify="right")
    table.add_column("licence")
    table.add_column("img2img", justify="center")
    for spec in specs:
        licence = spec.license.name
        if spec.license.commercial.value == "no":
            licence = f"[red]{licence}[/red]"
        if spec.license.gated:
            licence += " [yellow](token)[/yellow]"
        table.add_row(
            spec.key + (" *" if spec.key == "sdxl-lightning" else ""),
            f"{spec.default_width}x{spec.default_height}",
            # The step count at the default effort level, not the preset's raw
            # value: the ladder is what actually runs.
            str(spec.effort(DEFAULT_EFFORT).steps),
            f"{spec.download_gb:.1f} Go",
            f"{spec.weights_vram_gb:.1f} Go",
            licence,
            "oui" if spec.supports_img2img else "non",
        )
    console.print(table)
    console.print("[dim]* modele par defaut. 'imagegen models show <cle>' pour le detail.[/dim]")
    if not all_models:
        console.print("[dim]--all pour voir aussi les modeles optionnels.[/dim]")


@models_app.command("show")
def models_show(key: str = typer.Argument(..., help="Cle du modele.")) -> None:
    """Detaille un modele : licence, defauts, contraintes, portabilite."""
    from .models import get_model

    try:
        spec = get_model(key)
    except ImageGenError as exc:
        _fail(exc, False)
        return

    console.print(f"[bold cyan]{spec.key}[/bold cyan] - {spec.title}")
    console.print(spec.summary)
    console.print()

    table = Table(show_header=False, box=None)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("depot", spec.repo_id)
    table.add_row("famille", spec.family)
    licence = f"{spec.license.name} (usage commercial : {spec.license.commercial.value})"
    if spec.license.gated:
        licence += " - accord + token requis"
    table.add_row("licence", licence)
    if spec.license.notes:
        table.add_row("", f"[dim]{spec.license.notes}[/dim]")
    table.add_row("resolution", f"{spec.default_width}x{spec.default_height}")
    table.add_row("etapes", f"{spec.effort(DEFAULT_EFFORT).steps} (effort par defaut)")
    table.add_row("guidance", str(spec.default_guidance))
    if spec.scheduler:
        mandatory = " (impose)" if spec.scheduler.required else ""
        table.add_row("scheduler", spec.scheduler.kind.value + mandatory)
    if spec.lora:
        table.add_row("adaptateur", spec.lora.repo_id)
    table.add_row("telechargement", f"{spec.download_gb:.2f} Go")
    table.add_row("VRAM (poids)", f"{spec.weights_vram_gb:.2f} Go")
    table.add_row("strategie 8 Go", spec.tier.value)
    table.add_row("img2img", "oui" if spec.supports_img2img else "non")
    table.add_row("vitesse", spec.speed_note or "-")
    portable = []
    if spec.portable.openvino_repo:
        portable.append(f"OpenVINO: {spec.portable.openvino_repo}")
    if spec.portable.openvino_int8_repo:
        portable.append(f"OpenVINO int8: {spec.portable.openvino_int8_repo}")
    if spec.portable.onnx_repo:
        portable.append(f"ONNX: {spec.portable.onnx_repo}")
    table.add_row("portabilite", "\n".join(portable) if portable else "aucun artefact converti")
    console.print(table)

    console.print()
    ladder = Table(title="Echelle d'effort", title_justify="left")
    ladder.add_column("niveau", style="bold cyan")
    ladder.add_column("etapes", justify="right")
    ladder.add_column("guidage", justify="right")
    ladder.add_column("adaptateur")
    ladder.add_column("affinage")
    for level in Effort:
        profile = spec.effort(level)
        if profile.drop_lora:
            adapter = "[dim]aucun (modele de base)[/dim]"
        elif profile.lora is not None:
            adapter = profile.lora.weight_name or profile.lora.repo_id
        elif spec.lora is not None:
            adapter = spec.lora.weight_name or spec.lora.repo_id
        else:
            adapter = "-"
        if profile.refines:
            effective = int(profile.refine_steps * profile.refine_strength)
            scale = f" a {profile.refine_scale:g}x" if profile.refine_scale > 1 else ""
            refine = f"{effective} etapes{scale}"
        else:
            refine = "-"
        guidance = profile.guidance if profile.guidance is not None else spec.default_guidance
        marker = " [green](defaut)[/green]" if level is DEFAULT_EFFORT else ""
        ladder.add_row(level.value + marker, str(profile.steps), f"{guidance:g}", adapter, refine)
    console.print(ladder)

    if spec.warnings:
        console.print()
        for warning in spec.warnings:
            console.print(f"[yellow]![/yellow] {escape(warning)}")


# --------------------------------------------------------------------------- #
# devices / backends / doctor
# --------------------------------------------------------------------------- #

@app.command()
def devices(
    as_json: bool = typer.Option(False, "--json", help="Sortie machine."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Affiche les proprietes brutes."),
) -> None:
    """Liste les peripheriques de calcul detectes (GPU, NPU, CPU)."""
    from .hardware.detect import detect_devices

    found = detect_devices()
    if as_json:
        console.print_json(
            json.dumps(
                [
                    {
                        "id": d.id,
                        "selector": d.short_id,
                        "kind": d.kind.value,
                        "vendor": d.vendor.value,
                        "name": d.name,
                        "memory_gb": d.total_memory_gb,
                        "dynamic_shapes": d.supports_dynamic_shapes,
                        "fp16": d.supports_fp16,
                        "bf16": d.supports_bf16,
                        "notes": list(d.notes),
                    }
                    for d in found
                ],
                ensure_ascii=False,
            )
        )
        return

    table = Table(title="Peripheriques detectes")
    table.add_column("selecteur", style="bold cyan")
    table.add_column("type")
    table.add_column("nom")
    table.add_column("memoire", justify="right")
    table.add_column("formes")
    table.add_column("precisions")
    for index, dev in enumerate(found):
        memory = f"{dev.total_memory_gb:.1f} Go" if dev.total_memory_gb else "partagee"
        precisions = ", ".join(
            [p for p, ok in (("fp16", dev.supports_fp16), ("bf16", dev.supports_bf16)) if ok]
        ) or "fp32"
        table.add_row(
            dev.short_id + (" [green](auto)[/green]" if index == 0 else ""),
            dev.kind.value,
            dev.name[:42],
            memory,
            "libres" if dev.supports_dynamic_shapes else "[yellow]fixes[/yellow]",
            precisions,
        )
    console.print(table)

    for dev in found:
        for note in dev.notes:
            console.print(f"[dim]{dev.short_id}: {note}[/dim]")
    if verbose:
        for dev in found:
            if dev.properties:
                console.print(f"[dim]{dev.short_id} -> {dev.properties}[/dim]")


@app.command()
def backends() -> None:
    """Liste les backends d'inference et pourquoi ils sont, ou non, utilisables."""
    from .backends import backend_report

    table = Table(title="Backends")
    table.add_column("nom", style="bold cyan")
    table.add_column("etat")
    table.add_column("format")
    table.add_column("detail")
    for cls, availability in backend_report():
        if availability.available and availability.implemented:
            state = "[green]pret[/green]"
        elif availability.available:
            state = "[yellow]experimental[/yellow]"
        else:
            state = "[red]indisponible[/red]"
        # Install hints contain pip extras in square brackets, which rich would
        # otherwise swallow as markup tags.
        detail = escape(availability.reason or cls.description)
        if not availability.available and availability.install_hint:
            detail += f"\n[dim]{escape(availability.install_hint)}[/dim]"
        table.add_row(cls.name, state, cls.model_format, detail)
    console.print(table)


@app.command()
def doctor(
    smoke: bool = typer.Option(
        False, "--smoke", help="Lance une vraie generation minimale (telecharge ~2,7 Go)."
    ),
) -> None:
    """Diagnostique l'installation : runtimes, pilotes, noyaux GPU, cache."""
    from .backends import backend_report
    from .backends.torch_diffusers import TorchDiffusersBackend
    from .hardware.detect import detect_devices, runtime_statuses

    ok = True
    console.print(f"[bold]imagegen {__version__}[/bold]  -  Python {sys.version.split()[0]}")
    console.print()

    console.print("[bold]Runtimes[/bold]")
    for status in runtime_statuses():
        mark = "[green]ok[/green]" if status.installed else "[dim]absent[/dim]"
        line = f"  {mark} {status.runtime.value}"
        if status.version:
            line += f" ({status.version})"
        if status.device_count:
            line += f" - {status.device_count} peripherique(s)"
        console.print(line)
        if status.error:
            console.print(f"      [red]{escape(status.error)}[/red]")
        elif not status.installed and status.install_hint:
            console.print(f"      [dim]{escape(status.install_hint)}[/dim]")

    console.print()
    console.print("[bold]Peripheriques[/bold]")
    found = detect_devices()
    for dev in found:
        console.print(f"  {dev.short_id}: {escape(dev.label())}")

    console.print()
    console.print("[bold]Noyaux GPU[/bold]")
    # torch.cuda.is_available() returns True on a wheel built without this
    # card's architecture; the failure would otherwise appear as "no kernel
    # image is available" deep inside a denoising loop.
    cuda_devices = [d for d in found if d.runtime.value == "torch-cuda"]
    if not cuda_devices:
        console.print("  [dim]aucun GPU NVIDIA detecte[/dim]")
    for dev in cuda_devices:
        problem = TorchDiffusersBackend.check_cuda_kernels(dev)
        if problem:
            ok = False
            console.print(f"  [red]probleme[/red] {problem}")
        else:
            major, minor = dev.compute_capability or (0, 0)
            console.print(f"  [green]ok[/green] {dev.name} - noyaux sm_{major}{minor} presents")

    console.print()
    console.print("[bold]Backends[/bold]")
    for cls, availability in backend_report():
        if availability.available and availability.implemented:
            console.print(f"  [green]ok[/green] {cls.name}")
        elif availability.available:
            console.print(f"  [yellow]experimental[/yellow] {cls.name} - {escape(availability.reason)}")
        else:
            console.print(f"  [dim]absent[/dim] {cls.name} - {escape(availability.reason)}")

    console.print()
    console.print("[bold]Cache et stockage[/bold]")
    for line in _cache_report():
        console.print(f"  {line}")

    if smoke:
        console.print()
        console.print("[bold]Test de generation[/bold]")
        ok = _smoke_test() and ok

    console.print()
    console.print("[green]Installation fonctionnelle.[/green]" if ok else "[red]Problemes detectes (voir ci-dessus).[/red]")
    if not ok:
        raise typer.Exit(1)


@app.command()
def download(
    key: str = typer.Argument(..., help="Cle du modele a pre-telecharger."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
) -> None:
    """Telecharge les poids d'un modele sans generer d'image."""
    from huggingface_hub import snapshot_download

    from .models import get_model

    try:
        spec = get_model(key)
    except ImageGenError as exc:
        _fail(exc, False)
        return

    console.print(f"Telechargement de [bold]{spec.repo_id}[/bold] (~{spec.download_gb:.1f} Go)...")
    targets = [(spec.repo_id, None)]
    if spec.vae_repo:
        targets.append((spec.vae_repo, None))
    # Every adapter the effort ladder can select, not just the preset's own:
    # otherwise --effort draft or balanced hits the network on a machine the
    # user believed was fully provisioned.
    adapters: list[tuple[str, str | None]] = []
    for candidate in [spec.lora] + [spec.effort(level).lora for level in Effort]:
        if candidate is None:
            continue
        entry = (candidate.repo_id, candidate.weight_name)
        if entry not in adapters:
            adapters.append(entry)
    targets.extend(adapters)

    try:
        for repo, weight_name in targets:
            patterns = [weight_name] if weight_name else None
            path = snapshot_download(
                repo,
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=patterns,
                # The safety checker is ~1.2 GB and is never loaded.
                ignore_patterns=["*.ckpt", "*non_ema*", "safety_checker/*"] if patterns is None else None,
            )
            console.print(f"  [green]ok[/green] {repo} -> {path}")
    except Exception as exc:
        error_console.print(f"[red]Echec du telechargement :[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def version() -> None:
    """Affiche les versions installees."""
    console.print(f"imagegen {__version__}")
    for module in ("torch", "diffusers", "transformers", "huggingface_hub", "openvino", "onnxruntime"):
        try:
            import importlib.metadata as meta

            console.print(f"  {module} {meta.version(module)}")
        except Exception:
            console.print(f"  [dim]{module} absent[/dim]")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_choice(value: str, allowed: set[str], flag: str) -> str:
    normalised = value.strip().lower()
    if normalised not in allowed:
        raise ConfigurationError(
            f"Valeur '{value}' invalide pour {flag}.",
            hint="Valeurs possibles : " + ", ".join(sorted(allowed)) + ".",
        )
    return normalised


def _parse_effort(value: str) -> Effort:
    try:
        return Effort(value.strip().lower())
    except ValueError:
        raise ConfigurationError(
            f"Niveau d'effort '{value}' inconnu.",
            hint="Valeurs possibles : " + ", ".join(level.value for level in Effort) + ".",
        ) from None


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _fail(exc: ImageGenError, debug: bool) -> None:
    if debug:
        raise exc
    error_console.print(f"[red]Erreur :[/red] {escape(exc.message)}")
    if exc.hint:
        error_console.print(f"[yellow]->[/yellow] {escape(exc.hint)}")
    raise typer.Exit(1)


def _install_interrupt_handler():
    """Turn Ctrl-C into a cooperative cancel, checked between denoising steps."""
    state = {"cancelled": False}

    def handler(signum: int, frame: object) -> None:
        state["cancelled"] = True

    # Fails outside the main thread, where the default handler still applies.
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGINT, handler)
    return lambda: state["cancelled"]


class _NullProgress:
    def __enter__(self):
        return (None, None)

    def __exit__(self, *exc: object) -> None:
        return None


def _progress(disabled: bool):
    if disabled:
        return _NullProgress()

    class _Ctx:
        def __enter__(self):
            self.bar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=True,
            )
            self.bar.start()
            self.task = self.bar.add_task("debruitage", total=None)
            return (self.bar, self.task)

        def __exit__(self, *exc: object) -> None:
            self.bar.stop()

    return _Ctx()


def _print_plan(generator, plan) -> None:
    info = generator.describe()
    spec, resolved = plan.spec, plan.request
    mode = "image-to-image" if spec.mode.value == "img2img" else "text-to-image"
    profile = generator.model.effort(generator.effort)
    extras = ""
    if profile.refines:
        extras = f" + affinage {profile.refine_steps} etapes"
    console.print(
        f"[bold]{info['model']}[/bold] [dim]({spec.effort.value})[/dim] - {mode} - "
        f"{spec.width}x{spec.height} - {resolved.steps} etapes{extras} - "
        f"cfg {resolved.guidance_scale:g} - {info['device_name']} "
        f"([cyan]{info['device']}[/cyan])"
    )
    # Adjustments made on the user's behalf are shown before the work starts,
    # not buried in the summary afterwards.
    for warning in plan.warnings:
        console.print(f"[yellow]![/yellow] {escape(warning)}")


def _print_result(result, skip: set[str] | None = None) -> None:
    for warning in result.warnings:
        if skip and warning in skip:
            continue
        console.print(f"[yellow]![/yellow] {escape(warning)}")
    if result.load_s:
        console.print(f"[dim]chargement : {result.load_s:.1f} s[/dim]")
    console.print(
        f"[green]{len(result.images)} image(s)[/green] en {result.duration_s:.1f} s "
        f"({result.seconds_per_image:.1f} s/image)"
    )
    for generated, path in zip(result.images, result.paths, strict=False):
        console.print(f"  {path}  [dim]graine {generated.seed}[/dim]")


def _cache_report() -> list[str]:
    lines: list[str] = []
    try:
        from huggingface_hub import constants, scan_cache_dir

        info = scan_cache_dir()
        lines.append(
            f"cache Hugging Face : {info.size_on_disk_str} "
            f"({len(info.repos)} depots) dans {constants.HF_HUB_CACHE}"
        )
        if info.warnings:
            lines.append(f"[yellow]{len(info.warnings)} avertissement(s) de cache[/yellow]")
        if info.incomplete_files:
            # Left behind by an interrupted download; they are resumed, not
            # wasted, but a user seeing the disk fill deserves to know.
            lines.append(
                f"[yellow]{len(info.incomplete_files)} telechargement(s) incomplet(s)[/yellow] "
                f"({info.incomplete_size_on_disk / 1e9:.1f} Go)"
            )
    except Exception as exc:
        lines.append(f"[dim]cache illisible : {escape(str(exc))}[/dim]")

    try:
        import shutil

        usage = shutil.disk_usage(Path.cwd().anchor)
        free_gb = usage.free / (1024**3)
        marker = "[red]" if free_gb < 15 else "[green]"
        lines.append(f"{marker}{free_gb:.0f} Go libres[/]  sur {Path.cwd().anchor}")
    except Exception:  # pragma: no cover
        pass

    token = os.environ.get("HF_TOKEN")
    if token:
        lines.append("jeton Hugging Face : present (HF_TOKEN)")
    else:
        try:
            from huggingface_hub import get_token

            lines.append(
                "jeton Hugging Face : "
                + ("present" if get_token() else "absent (requis pour les depots sous accord)")
            )
        except Exception:  # pragma: no cover
            pass
    return lines


def _smoke_test() -> bool:
    """Generate one tiny image, end to end.

    diffusers 0.40 on recent torch and Blackwell was not verified upstream, so
    the only trustworthy check is running the real thing at a size where it
    costs a couple of seconds.
    """
    from .backends import LoadOptions
    from .generation import ImageGenerator

    try:
        generator = ImageGenerator.create(model="sd15", device="auto", options=LoadOptions())
        request = GenerationRequest(prompt="a red cube on a table", steps=2, width=256, height=256, seed=0)
        result = generator.generate(request)
        image = result.images[0].image
        console.print(
            f"  [green]ok[/green] image {image.size[0]}x{image.size[1]} en {result.duration_s:.1f} s "
            f"(chargement {result.load_s:.1f} s)"
        )
        generator.unload()
        return True
    except ImageGenError as exc:
        console.print(f"  [red]echec[/red] {exc.message}")
        if exc.hint:
            console.print(f"      [dim]{exc.hint}[/dim]")
        return False
    except Exception as exc:  # pragma: no cover - surfaced as a diagnostic
        console.print(f"  [red]echec[/red] {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
