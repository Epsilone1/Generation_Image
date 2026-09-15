"""Language detection for prompts.

The detector decides whether to translate, so both its failure directions cost
something real: missing a French prompt means CLIP silently drops words, and
flagging an English one means translating text that was already fine. It is
deliberately biased towards the second being rare.

No model is loaded here - detection is pure text.
"""

from __future__ import annotations

import pytest

from imagegen.generation.prompts import detect_language, prepare_prompts

FRENCH = [
    "un renard roux dans une forêt enneigée",
    "un phare dans la tempête, peinture à l'huile",
    "portrait d'un vieux pêcheur au visage buriné",
    # Unaccented typing is the case that translates worst, so it must be caught.
    "un verre d'eau et une cle en laiton sur une table en bois sombre",
    "une ruelle pavee dans une vieille ville europeenne au crepuscule",
    "un chat roux sur le canape",
]

ENGLISH = [
    "a red fox standing in a snowy forest, wildlife photography",
    "close-up portrait of an elderly fisherman, natural light",
    "a glass of water and a brass key on a dark wooden table",
    "cinematic still of a lighthouse in a storm, oil painting",
    "highly detailed concept art of a floating city, volumetric lighting",
    "a cat",
]


@pytest.mark.parametrize("prompt", FRENCH)
def test_french_prompts_are_detected(prompt):
    assert detect_language(prompt) == "fr"


@pytest.mark.parametrize("prompt", ENGLISH)
def test_english_prompts_are_left_alone(prompt):
    assert detect_language(prompt) is None


def test_a_stray_french_word_does_not_trigger_translation():
    """One borrowed word is not a French prompt."""
    assert detect_language("a portrait in the style of art nouveau, la belle epoque") is None


def test_a_short_unambiguous_prompt_is_still_detected():
    """"un chat" is French, and translating it to "a cat" is the right move."""
    assert detect_language("un chat") == "fr"


def test_a_single_word_is_never_guessed():
    """One word carries too little signal to risk translating."""
    assert detect_language("chat") is None
    assert detect_language("fox") is None


def test_spanish_and_italian_are_recognised():
    assert detect_language("un gato rojo en un bosque nevado con luz calida") == "es"
    assert detect_language("un gatto rosso in una foresta innevata con luce calda") == "it"


def test_never_mode_leaves_the_prompt_untouched():
    prompt, negative = prepare_prompts(
        "un renard roux dans une forêt", "flou", mode="never"
    )
    assert prompt.text == "un renard roux dans une forêt"
    assert not prompt.translated
    assert negative is not None and negative.text == "flou"


def test_english_prompt_is_not_translated_in_auto_mode():
    """No model is loaded at all when there is nothing to translate."""
    prompt, _ = prepare_prompts("a red fox in a snowy forest", None, mode="auto")
    assert prompt.text == "a red fox in a snowy forest"
    assert not prompt.translated
    assert prompt.source_language is None


def test_unavailable_translator_degrades_to_a_warning():
    """A missing translation must cost adherence, never the generation."""

    class _Broken:
        def translate(self, texts, language):
            return None

    prompt, _ = prepare_prompts(
        "un renard roux dans une forêt enneigée", None, mode="auto", translator=_Broken()
    )
    assert prompt.text == "un renard roux dans une forêt enneigée"
    assert not prompt.translated
    assert "anglais" in prompt.note


def test_translation_is_reported_not_silent():
    class _Fake:
        def translate(self, texts, language):
            return ["a red fox in a snowy forest"] + ["blurry"] * (len(texts) - 1)

    prompt, negative = prepare_prompts(
        "un renard roux dans une forêt enneigée", "flou", mode="auto", translator=_Fake()
    )
    assert prompt.translated
    assert prompt.text == "a red fox in a snowy forest"
    assert prompt.original == "un renard roux dans une forêt enneigée"
    assert "traduit" in prompt.note
    assert negative is not None and negative.text == "blurry"
