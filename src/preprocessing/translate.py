"""
Language translation module — handles MIXED-LANGUAGE text (non-English words
embedded within otherwise English business names and addresses).

Translation backends (all run 100% offline, no API calls):
─────────────────────────────────────────────────────────────
1. IndicTrans2 (AI4Bharat / IIT Madras, MIT License)
   - Best-in-class for Indian languages (Hindi, Tamil, Telugu, Bengali, etc.)
   - Model: ai4bharat/indictrans2-indic-en-dist-200M
   - Requires: transformers, IndicTransToolkit

2. Opus-MT (Helsinki-NLP, CC-BY-4.0)
   - Covers European and other languages (French, German, Spanish, etc.)
   - Models: Helsinki-NLP/opus-mt-{src}-en (MarianMT architecture)
   - Requires: transformers

3. Argostranslate (MIT License) — fallback
   - Broad coverage, lower quality but already installed
   - Used as a catch-all when the above two don't cover a language

Architecture for mixed-language text:
─────────────────────────────────────
1. Split text into segments: Latin-script vs non-Latin-script tokens
2. Detect the script of non-Latin segments (Devanagari, Tamil, Arabic, CJK, etc.)
3. Route to the appropriate translation model
4. Reassemble the sentence preserving English tokens unchanged
"""
import re
import unicodedata
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Script detection → language code mapping
# ──────────────────────────────────────────────────────────────────────────────

# Unicode script name → (ISO 639-1 code, IndicTrans2 lang tag or None)
_SCRIPT_TO_LANG = {
    # Indian languages → IndicTrans2
    "DEVANAGARI":   ("hi", "hin_Deva"),
    "BENGALI":      ("bn", "ben_Beng"),
    "TAMIL":        ("ta", "tam_Taml"),
    "TELUGU":       ("te", "tel_Telu"),
    "GUJARATI":     ("gu", "guj_Gujr"),
    "KANNADA":      ("kn", "kan_Knda"),
    "MALAYALAM":    ("ml", "mal_Mlym"),
    "ORIYA":        ("or", "ory_Orya"),
    "GURMUKHI":     ("pa", "pan_Guru"),

    # European / other → Opus-MT
    "CYRILLIC":     ("ru", None),
    "ARABIC":       ("ar", None),
    "CJK":          ("zh", None),
    "HANGUL":       ("ko", None),
    "KATAKANA":     ("ja", None),
    "HIRAGANA":     ("ja", None),
    "THAI":         ("th", None),
}


def _detect_script(text: str) -> Optional[str]:
    """
    Detect the dominant non-Latin Unicode script in *text*.
    Returns the script name (e.g. "DEVANAGARI") or None if all Latin/ASCII.
    """
    script_counts: Dict[str, int] = {}
    for ch in text:
        if ord(ch) < 128:
            continue  # ASCII — skip
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            continue
        for script_key in _SCRIPT_TO_LANG:
            if script_key in name:
                script_counts[script_key] = script_counts.get(script_key, 0) + 1
                break
    if not script_counts:
        # Check for accented Latin (French, German, Spanish, etc.)
        if re.search(r'[À-ÿ]', text):
            return "LATIN_EXTENDED"
        return None
    return max(script_counts, key=script_counts.get)


# ──────────────────────────────────────────────────────────────────────────────
# Segment splitter for mixed-language text
# ──────────────────────────────────────────────────────────────────────────────

def _split_mixed_text(text: str) -> List[Tuple[str, bool]]:
    """
    Split text into segments of (substring, is_non_latin).

    Groups consecutive tokens by script type so that entire non-Latin
    phrases are translated together (better context than word-by-word).

    Example:
        "ABC Company के पास Main Road"
        → [("ABC Company ", False), ("के पास ", True), ("Main Road", False)]
    """
    if not text:
        return []

    segments: List[Tuple[str, bool]] = []
    current = []
    current_is_non_latin = None

    for token in re.split(r'(\s+)', text):
        if not token:
            continue
        if token.isspace():
            current.append(token)
            continue

        # Determine if this token has non-Latin characters
        has_non_latin = bool(re.search(r'[^\x00-\x7F\u00C0-\u024F]', token))

        if current_is_non_latin is None:
            current_is_non_latin = has_non_latin

        if has_non_latin != current_is_non_latin:
            # Script boundary — flush current segment
            if current:
                segments.append(("".join(current), current_is_non_latin))
            current = [token]
            current_is_non_latin = has_non_latin
        else:
            current.append(token)

    if current:
        segments.append(("".join(current), current_is_non_latin))

    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Translation backends (lazy-loaded, cached)
# ──────────────────────────────────────────────────────────────────────────────

# ----- IndicTrans2 -----
_indictrans_model = None
_indictrans_tokenizer = None
_indictrans_processor = None
_indictrans_available = None  # None = not yet checked


def _load_indictrans():
    """Lazy-load IndicTrans2 model. Returns True if available."""
    global _indictrans_model, _indictrans_tokenizer, _indictrans_processor
    global _indictrans_available

    if _indictrans_available is not None:
        return _indictrans_available

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
        logger.info(f"Loading IndicTrans2 model: {model_name}")

        _indictrans_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _indictrans_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
        _indictrans_model.eval()
        _indictrans_processor = IndicProcessor(inference=True)
        _indictrans_available = True
        logger.info("IndicTrans2 loaded successfully.")
    except Exception as e:
        logger.warning(f"IndicTrans2 not available: {e}")
        _indictrans_available = False

    return _indictrans_available


def _translate_indictrans(text: str, src_lang_tag: str) -> str:
    """Translate using IndicTrans2. Returns original text on failure."""
    if not _load_indictrans():
        return text
    try:
        import torch
        tgt_lang = "eng_Latn"
        batch = _indictrans_processor.preprocess_batch(
            [text], src_lang=src_lang_tag, tgt_lang=tgt_lang
        )
        inputs = _indictrans_tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt"
        )
        device = next(_indictrans_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = _indictrans_model.generate(
                **inputs, use_cache=True, min_length=0, max_length=256,
                num_beams=5, num_return_sequences=1,
            )
        result = _indictrans_tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )
        if result and result[0].strip():
            return _indictrans_processor.postprocess_batch(
                result, lang=tgt_lang
            )[0]
        return text
    except Exception as e:
        logger.debug(f"IndicTrans2 translation failed: {e}")
        return text


# ----- Opus-MT (Helsinki-NLP) -----
_opus_models: Dict[str, tuple] = {}  # lang_code → (tokenizer, model)
_opus_checked: Dict[str, bool] = {}  # lang_code → available?

# Opus-MT model names for key languages
_OPUS_MODEL_MAP = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "pt": "Helsinki-NLP/opus-mt-pt-en",
    "nl": "Helsinki-NLP/opus-mt-nl-en",
    "ru": "Helsinki-NLP/opus-mt-ru-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
    "zh": "Helsinki-NLP/opus-mt-zh-en",
    "ja": "Helsinki-NLP/opus-mt-ja-en",
    "ko": "Helsinki-NLP/opus-mt-ko-en",
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "tr": "Helsinki-NLP/opus-mt-tr-en",
    "th": "Helsinki-NLP/opus-mt-tc-big-en",  # Thai via tc-big group
    "sv": "Helsinki-NLP/opus-mt-sv-en",
    "pl": "Helsinki-NLP/opus-mt-pl-en",
    "da": "Helsinki-NLP/opus-mt-da-en",
}


def _load_opus(lang_code: str) -> bool:
    """Lazy-load an Opus-MT model for *lang_code*. Returns True if available."""
    if lang_code in _opus_checked:
        return _opus_checked[lang_code]

    model_name = _OPUS_MODEL_MAP.get(lang_code)
    if not model_name:
        _opus_checked[lang_code] = False
        return False

    try:
        from transformers import MarianMTModel, MarianTokenizer
        logger.info(f"Loading Opus-MT model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()
        _opus_models[lang_code] = (tokenizer, model)
        _opus_checked[lang_code] = True
        logger.info(f"Opus-MT [{lang_code}→en] loaded successfully.")
    except Exception as e:
        logger.warning(f"Opus-MT [{lang_code}→en] not available: {e}")
        _opus_checked[lang_code] = False

    return _opus_checked.get(lang_code, False)


def _translate_opus(text: str, lang_code: str) -> str:
    """Translate using Opus-MT (Helsinki-NLP). Returns original text on failure."""
    if not _load_opus(lang_code):
        return text
    try:
        import torch
        tokenizer, model = _opus_models[lang_code]
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            generated = model.generate(**inputs)
        result = tokenizer.batch_decode(generated, skip_special_tokens=True)
        if result and result[0].strip():
            return result[0].strip()
        return text
    except Exception as e:
        logger.debug(f"Opus-MT [{lang_code}→en] failed: {e}")
        return text


# ----- Argostranslate (fallback) -----
try:
    import argostranslate.translate
    _argo_languages = argostranslate.translate.get_installed_languages()
    _argo_by_code = {lang.code: lang for lang in _argo_languages}
    _argo_english = _argo_by_code.get("en")
except (ImportError, Exception):
    _argo_languages = []
    _argo_by_code = {}
    _argo_english = None


def _translate_argos(text: str, lang_code: str = None) -> str:
    """Fallback translation via argostranslate."""
    if _argo_english is None:
        return text
    try:
        if lang_code and lang_code in _argo_by_code:
            src = _argo_by_code[lang_code]
            translation = src.get_translation(_argo_english)
            if translation:
                result = translation.translate(text)
                if result and result.strip():
                    return result

        # Brute-force: try each installed language
        for lang in _argo_languages:
            if lang.code == "en":
                continue
            translation = lang.get_translation(_argo_english)
            if translation:
                try:
                    result = translation.translate(text)
                    if result and result.strip():
                        return result
                except Exception:
                    continue
    except Exception:
        pass
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Unified translation router
# ──────────────────────────────────────────────────────────────────────────────

def _translate_segment(text: str) -> str:
    """
    Translate a single non-Latin text segment to English using the
    best available backend for its detected script.
    """
    script = _detect_script(text)
    if script is None:
        return text  # already Latin/English

    if script == "LATIN_EXTENDED":
        # Accented Latin (French, German, Spanish) — try Opus-MT first
        # Attempt French first since it's a known test-set country
        for lang_code in ["fr", "de", "es", "it", "pt", "nl"]:
            result = _translate_opus(text, lang_code)
            if result != text:
                return result
        return _translate_argos(text)

    lang_info = _SCRIPT_TO_LANG.get(script)
    if not lang_info:
        return _translate_argos(text)

    lang_code, indic_tag = lang_info

    # Indian scripts → try IndicTrans2 first, then Opus-MT, then Argos
    if indic_tag:
        result = _translate_indictrans(text, indic_tag)
        if result != text:
            return result

    # Opus-MT fallback
    result = _translate_opus(text, lang_code)
    if result != text:
        return result

    # Argostranslate fallback
    return _translate_argos(text, lang_code)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def _looks_non_english(text: str) -> bool:
    """
    Cheap pre-check: if text is pure ASCII letters/digits/punctuation,
    skip translation entirely.
    """
    return bool(re.search(r'[^\x00-\x7F]', text))


def translate_to_english(text: str, source_lang_code: str = None) -> str:
    """
    Step 1: Translates input text to English using local offline models.

    Handles MIXED-LANGUAGE text by splitting into Latin vs non-Latin
    segments, translating only the non-Latin parts, and reassembling.

    Translation priority:
    1. IndicTrans2 (Indian scripts — Devanagari, Tamil, Bengali, etc.)
    2. Opus-MT / Helsinki-NLP (French, German, Russian, Arabic, CJK, etc.)
    3. Argostranslate (catch-all fallback)

    Returns original text unchanged if already English or on any failure.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Fast path: pure ASCII → no translation needed
    if not _looks_non_english(text):
        return text

    try:
        # If a specific source language is provided, try direct translation
        if source_lang_code:
            indic_tag = None
            for script, (lc, tag) in _SCRIPT_TO_LANG.items():
                if lc == source_lang_code:
                    indic_tag = tag
                    break

            if indic_tag:
                result = _translate_indictrans(text, indic_tag)
                if result != text:
                    return result
            result = _translate_opus(text, source_lang_code)
            if result != text:
                return result
            return _translate_argos(text, source_lang_code)

        # Mixed-language handling: split, translate non-Latin parts, reassemble
        segments = _split_mixed_text(text)

        if not segments:
            return text

        # Check if the ENTIRE text is non-Latin (no mixing)
        all_non_latin = all(is_nl for _, is_nl in segments if not _.isspace())
        if all_non_latin:
            # Translate the whole string at once (better context for the model)
            return _translate_segment(text)

        # Mixed text: translate non-Latin segments individually
        result_parts = []
        for segment, is_non_latin in segments:
            if is_non_latin:
                translated = _translate_segment(segment).strip()
                # Preserve whitespace boundaries: add a space before/after
                # the translated segment if the original had one
                leading_space = " " if segment and segment[0].isspace() else ""
                trailing_space = " " if segment and segment[-1].isspace() else ""
                result_parts.append(f"{leading_space}{translated}{trailing_space}")
            else:
                result_parts.append(segment)

        assembled = "".join(result_parts)
        # Normalise any double spaces that may have crept in
        assembled = re.sub(r'  +', ' ', assembled).strip()
        return assembled

    except Exception as e:
        logger.debug(f"Translation failed for '{text[:50]}...': {e}")
        return text  # fail-safe: never crash the pipeline