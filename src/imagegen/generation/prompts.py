"""Prompt preparation, and why a French prompt needs it.

SD 1.5 and SDXL encode prompts with CLIP text encoders trained almost entirely
on English. A French prompt does not fail - it quietly loses the words CLIP has
no representation for. Measured on this project's benchmark with SDXL-Lightning:
"un verre d'eau et une clé en laiton sur une table en bois sombre" produced two
glasses and **no key at all**, while the same prompt in English produced the
glass and the brass key correctly. Nothing warned; the image was simply not what
was asked for.

So a non-English prompt is translated before it reaches the encoders. The
translation is shown, never applied silently: machine translation makes its own
mistakes ("visage buriné" comes back as "burin face"), and the user needs to see
what was actually generated from.

Accents matter more than they look: "tempête" translates to "storm", while the
unaccented "tempete" comes back as "temple". The detector therefore treats an
unaccented French prompt as French too, and the CLI says so.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Marian models are small (~300 MB) and fast enough on the CPU that they never
#: compete with the diffusion pipeline for VRAM.
_MODELS: dict[str, str] = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
}

#: Function words that are common in the language and rare in an English prompt.
#: Matching on these rather than on accents catches unaccented typing, which is
#: exactly the case that translates worst.
_MARKERS: dict[str, frozenset[str]] = {
    "fr": frozenset(
        """un une des du de la le les dans sur sous avec sans pour vers chez
        et ou mais donc puis tres plus moins tout toute tous toutes
        au aux ce cet cette ses son sa leur leurs qui que quoi dont
        est sont etre avoir fait faire vu vue entre pendant apres avant
        peinture dessin photographie lumiere couleur ciel mer foret ville
        homme femme enfant chat chien maison arbre fleur nuit jour""".split()
    ),
    "es": frozenset(
        """un una unos unas el la los las en sobre bajo con sin para hacia
        y o pero muy mas menos todo toda todos todas del al que quien
        es son ser estar hecho hacer entre durante despues antes
        pintura dibujo fotografia luz color cielo mar bosque ciudad""".split()
    ),
    "de": frozenset(
        """der die das ein eine einen einem eines und oder aber sehr mehr
        mit ohne fur auf unter uber zwischen wahrend nach vor
        ist sind sein haben gemacht machen alle alles
        malerei zeichnung fotografie licht farbe himmel meer wald stadt""".split()
    ),
    "it": frozenset(
        """un uno una il lo la i gli le di del della nel sul con senza per
        e o ma molto piu meno tutto tutta tutti tutte che chi
        e sono essere avere fatto fare tra durante dopo prima
        pittura disegno fotografia luce colore cielo mare foresta citta""".split()
    ),
}

#: Words that are French/Spanish/Italian markers *and* ordinary English, so they
#: must not count on their own.
_AMBIGUOUS = frozenset({"a", "e", "i", "o", "no", "son", "die", "man", "an", "the"})

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    """The text actually sent to the model, plus what it came from."""

    text: str
    original: str
    source_language: str | None = None
    translated: bool = False
    note: str = ""


def _words(text: str) -> list[str]:
    stripped = unicodedata.normalize("NFKD", text.lower())
    ascii_text = stripped.encode("ascii", "ignore").decode("ascii")
    return _WORD.findall(ascii_text)


def detect_language(text: str) -> str | None:
    """Guess the prompt's language, or ``None`` when it looks like English.

    A deliberately conservative marker count rather than a dependency: prompts
    are short, and the cost of a wrong guess is translating an English prompt,
    which is worse than doing nothing.
    """
    words = [w for w in _words(text) if w not in _AMBIGUOUS]
    if len(words) < 2:
        return None

    scores = {
        language: sum(1 for word in words if word in markers)
        for language, markers in _MARKERS.items()
    }
    best = max(scores, key=lambda language: scores[language])
    hits = scores[best]
    # At least two marker words, and at least a fifth of the prompt: one stray
    # "le" in an English prompt must not trigger a translation.
    if hits >= 2 and hits / len(words) >= 0.2:
        return best
    return None


class PromptTranslator:
    """Lazily-loaded Marian translation into English.

    Never fatal: if the model cannot be fetched or loaded, the original prompt is
    used and the reason is reported. A missing translation should cost prompt
    adherence, not the whole generation.
    """

    def __init__(self, cache_dir: str | None = None, local_files_only: bool = False) -> None:
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._loaded: dict[str, tuple[object, object]] = {}

    def _load(self, language: str) -> tuple[object, object] | None:
        if language in self._loaded:
            return self._loaded[language]
        repo = _MODELS.get(language)
        if repo is None:
            return None
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                repo, cache_dir=self.cache_dir, local_files_only=self.local_files_only
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                repo, cache_dir=self.cache_dir, local_files_only=self.local_files_only
            ).eval()
        except Exception as exc:
            logger.debug("Traduction indisponible pour %s : %s", language, exc)
            return None
        self._loaded[language] = (tokenizer, model)
        return self._loaded[language]

    def translate(self, texts: list[str], language: str) -> list[str] | None:
        """Translate a batch into English, or ``None`` if it cannot be done."""
        loaded = self._load(language)
        if loaded is None:
            return None
        tokenizer, model = loaded
        try:
            import torch

            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                output = model.generate(**batch, max_new_tokens=128, num_beams=4)
            return [t.strip() for t in tokenizer.batch_decode(output, skip_special_tokens=True)]
        except Exception as exc:
            logger.debug("Traduction en echec : %s", exc)
            return None


def prepare_prompts(
    prompt: str,
    negative_prompt: str | None,
    *,
    mode: str = "auto",
    translator: PromptTranslator | None = None,
) -> tuple[PreparedPrompt, PreparedPrompt | None]:
    """Resolve the prompts that will reach the text encoders.

    ``mode`` is ``"auto"`` (translate when a non-English prompt is detected),
    ``"always"`` or ``"never"``.
    """
    if mode == "never":
        return PreparedPrompt(prompt, prompt), (
            PreparedPrompt(negative_prompt, negative_prompt) if negative_prompt else None
        )

    language = detect_language(prompt)
    if mode == "always" and language is None:
        language = "fr"  # forced: assume the project's own language
    if language is None:
        return PreparedPrompt(prompt, prompt), (
            PreparedPrompt(negative_prompt, negative_prompt) if negative_prompt else None
        )

    translator = translator or PromptTranslator()
    texts = [prompt] + ([negative_prompt] if negative_prompt else [])
    result = translator.translate(texts, language)
    if result is None:
        note = (
            f"Prompt detecte comme non anglais ({language}) mais la traduction est indisponible. "
            "Les encodeurs CLIP de ce modele sont entraines en anglais : certains mots seront "
            "ignores. Installez le modele de traduction ou ecrivez le prompt en anglais."
        )
        return PreparedPrompt(prompt, prompt, language, False, note), (
            PreparedPrompt(negative_prompt, negative_prompt) if negative_prompt else None
        )

    translated_prompt = PreparedPrompt(
        text=result[0],
        original=prompt,
        source_language=language,
        translated=True,
        note=f"Prompt traduit ({language} -> en) : \"{result[0]}\"",
    )
    translated_negative = None
    if negative_prompt:
        translated_negative = PreparedPrompt(
            text=result[1],
            original=negative_prompt,
            source_language=language,
            translated=True,
        )
    return translated_prompt, translated_negative
